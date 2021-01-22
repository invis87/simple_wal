//! A simple write-ahead-logging crate.
//!
//! Features
//!  - Optimized for sequential reads & writes
//!  - Easy atomic log compaction
//!  - Advisory locking
//!  - CRC32 checksums
//!  - Range scans
//!  - Persistent log entry index
//!
//! The entire log is scanned through on startup in order to detect & clean interrupted
//! writes and determine the length of the log. It's recommended to compact the log when
//! old entries are no longer likely to be used.
//!
//! ## Usage:
//!
//! ```
//! use simple_wal::LogFile;
//!
//! let path = std::path::Path::new("./wal-log");
//!
//! {
//!     let mut log = LogFile::open(path).unwrap();
//!
//!     // write to log
//!     log.write(&mut b"log entry".to_vec()).unwrap();
//!     log.write(&mut b"foobar".to_vec()).unwrap();
//!     log.write(&mut b"123".to_vec()).unwrap();
//!    
//!     // flush to disk
//!     log.flush().unwrap();
//! }
//!
//! {
//!     let mut log = LogFile::open(path).unwrap();
//!
//!     // Iterate through the log
//!     let mut iter = log.iter(..).unwrap();
//!     assert_eq!(iter.next().unwrap().unwrap(), b"log entry".to_vec());
//!     assert_eq!(iter.next().unwrap().unwrap(), b"foobar".to_vec());
//!     assert_eq!(iter.next().unwrap().unwrap(), b"123".to_vec());
//!     assert!(iter.next().is_none());
//! }
//!
//! {
//!     let mut log = LogFile::open(path).unwrap();
//!
//!     // Compact the log
//!     log.compact(1).unwrap();
//!
//!     // Iterate through the log
//!     let mut iter = log.iter(..).unwrap();
//!     assert_eq!(iter.next().unwrap().unwrap(), b"foobar".to_vec());
//!     assert_eq!(iter.next().unwrap().unwrap(), b"123".to_vec());
//!     assert!(iter.next().is_none());
//! }
//!
//! # let _ = std::fs::remove_file(path);
//! ```
//!
//!
//! ## Log Format:
//!
//! ```txt
//! 00 01 02 03 04 05 06 07|08 09 10 11 12 13 14 15|.......|-4 -3 -2 -1|
//! -----------------------|-----------------------|-------|-----------|
//! starting index         |entry length           | entry | crc32     |
//! unsigned 64 bit int le |unsigned 64 bit int le | data  | 32bit, le |
//! ```
//!
//! Numbers are stored in little-endian format.
//!
//! The first 8 bytes in the WAL is the starting index.
//!
//! Each entry follows the following format:
//! 1. A 64 bit unsigned int for the entry size.
//! 2. The entry data
//! 3. A 32 bit crc32 checksum.

use async_trait::async_trait;
use crc32fast;
use futures::future::TryFutureExt;
use futures::Stream;
use pin_project::pin_project;
use std::ops::{Bound, RangeBounds};
use std::path::PathBuf;
use std::{convert::TryInto, io::Write};
use thiserror::Error;

use tokio::fs::File;
use tokio::io::{
    self, AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt, AsyncWrite, AsyncWriteExt, SeekFrom,
};

/// A write-ahead-log.
pub struct LogFile {
    file: File,
    path: PathBuf,

    /// The index of the first log entry stored
    first_index: u64,
    len: u64,
}

impl LogFile {
    /// The first entry in the log
    pub async fn first_entry<'l>(&'l mut self) -> Result<LogEntry<'l>, LogError> {
        if self.len == 0 {
            return Err(LogError::OutOfBounds);
        }

        // Seek past to position 8 (immediately after the starting index)
        self.file.seek(SeekFrom::Start(8)).await?;

        let index = self.first_index;

        Ok(LogEntry { log: self, index })
    }

    /// Seek to the given entry in the log
    pub async fn seek<'l>(&'l mut self, to_index: u64) -> Result<LogEntry<'l>, LogError> {
        self.first_entry().await?.seek(to_index).await
    }

    /// Returns the index/sequence number of the first entry in the log
    pub fn first_index(&self) -> u64 {
        self.first_index
    }

    /// Return if there are any entries in the log
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the index/sequence number of the last entry in the log
    pub fn last_index(&self) -> u64 {
        let last_index = self.first_index + self.len;
        if last_index > 0 {
            last_index - 1
        } else {
            0
        }
    }

    /// Iterate through the log
    pub async fn iter<'s, R: RangeBounds<u64>>(
        &'s mut self,
        range: R,
    ) -> Result<LogIterator<'s>, LogError> {
        if self.len == 0 {
            return Ok(LogIterator {
                next: None,
                last_index: self.first_index,
                future: None,
            });
        }

        let last_index = match range.end_bound() {
            Bound::Unbounded => self.last_index(),
            Bound::Included(x) if self.last_index() > *x => *x,
            Bound::Excluded(x) if self.last_index() > *x - 1 => *x - 1,
            _ => return Err(LogError::OutOfBounds),
        };

        let start = match range.start_bound() {
            Bound::Unbounded => self.first_entry().await?,
            Bound::Included(x) => self.seek(*x).await?,
            Bound::Excluded(x) => self.seek(*x + 1).await?,
        };

        Ok(LogIterator {
            next: Some(start),
            last_index,
            future: None,
        })
    }

    /// Write the given log entry to the end of the log
    pub async fn write<R: AsMut<[u8]>>(&mut self, entry: &mut R) -> io::Result<()> {
        let end_pos = self.file.seek(SeekFrom::End(0)).await?;

        let entry = entry.as_mut();

        let hash = {
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(entry);
            &mut hasher.finalize().to_le_bytes()
        };

        let result = self
            .file
            .write_all(&mut (entry.len() as u64).to_le_bytes())
            .await;
        let result = self.file.write_all(entry).await;
        let result = self.file.write_all(hash).await;

        if result.is_ok() {
            self.len += 1;
        } else {
            // Trim the data written.
            self.file.set_len(end_pos + 1).await?;
        }

        result
    }

    /// Flush writes to disk
    pub async fn flush(&mut self) -> io::Result<()> {
        self.file.flush().await
    }

    /// Open the log. Takes out an advisory lock.
    ///
    /// This is O(n): we have to iterate to the end of the log in order to clean interrupted writes and determine the length of the log
    pub async fn open<P: AsRef<std::path::Path>>(path: P) -> Result<LogFile, LogError> {
        let mut file = File::open(&path).await?;
        let path = path.as_ref().to_owned();

        let file_size = file.metadata().await?.len();
        let mut entries: u64 = 0;
        let mut first_index: u64 = 0;

        if file_size >= 8 {
            first_index = file.read_u64().await?;

            let mut pos = 8;

            while file_size - pos > 8 {
                let entry_data_len = file.read_u64().await? + 4; // 4 byte checksum

                if file_size - pos - 8 < entry_data_len {
                    // the entry was not fully written
                    break;
                }

                entries += 1;
                pos = file
                    .seek(SeekFrom::Current(entry_data_len.try_into().unwrap()))
                    .await?;
            }

            file.set_len(pos).await?;
        } else {
            file.write_all(&mut [0; 8][..]).await?;
            file.set_len(8).await?;
        }

        Ok(LogFile {
            path,
            file,
            first_index,
            len: entries,
        })
    }

    /// Compact the log, removing entries older than `new_start_index`.
    ///
    /// This is done by copying all entries `>= new_start_index` to a temporary file, than overriding the
    /// old log file once the copy is complete.
    ///
    /// Before compacting, the log is flushed.
    pub async fn compact(&mut self, new_start_index: u64) -> Result<(), LogError> {
        self.flush().await?;

        // Seek to the start index. This will also change the file cursor, allowing io::copy to correctly operate.
        self.seek(new_start_index).await?;

        let mut orig_file_path = self.path.clone();
        let temp_file_path = match orig_file_path.file_name() {
            Some(temp_filename) => {
                let new_name = format!(
                    "{}_temp_{}",
                    temp_filename.to_string_lossy(),
                    rand::random::<u32>()
                );
                orig_file_path.set_file_name(new_name);
                orig_file_path
            }
            None => {
                let mut tempdir_file_path = std::env::temp_dir().to_path_buf();
                tempdir_file_path.set_file_name(format!("log-{}", rand::random::<u32>()));
                tempdir_file_path
            }
        };

        println!("create new file: {:?}", temp_file_path.as_path().to_str());

        let mut new_file = File::open(temp_file_path.as_path()).await?;

        new_file
            .write_all(&mut new_start_index.to_le_bytes())
            .await?;
        tokio::io::copy(&mut self.file, &mut new_file).await?;

        std::fs::rename(temp_file_path, self.path.clone())?;
        self.file = new_file;

        self.len = self.len - (new_start_index - self.first_index);
        self.first_index = new_start_index;

        Ok(())
    }

    /// Clear all entries in the write-ahead-log and restart at the given index.
    pub async fn restart(&mut self, starting_index: u64) -> Result<(), LogError> {
        self.file.seek(SeekFrom::Start(0)).await?;
        self.file.write_all(&starting_index.to_le_bytes()).await?;
        self.file.set_len(8).await?;
        self.file.flush().await?;

        self.first_index = starting_index;
        self.len = 0;

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum LogError {
    #[error("Bad checksum")]
    BadChecksum,
    #[error("Out of bounds")]
    OutOfBounds,
    #[error("{0}")]
    IoError(
        #[source]
        #[from]
        io::Error,
    ),
    #[error("the log is locked")]
    AlreadyLocked,
}

// impl From<F::FileLockError> for LogError {
//     fn from(err: advisory_lock::FileLockError) -> Self {
//         match err {
//             advisory_lock::FileLockError::IOError(err) => LogError::IoError(err),
//             advisory_lock::FileLockError::AlreadyLocked => LogError::AlreadyLocked,
//         }
//     }
// }

/// An entry in the log.
///
/// Ownership of this struct represents that the file has been seeked to the
/// start of the log entry.
pub struct LogEntry<'l> {
    log: &'l mut LogFile,
    index: u64,
}

impl<'l> LogEntry<'l> {
    pub fn index(&self) -> u64 {
        self.index
    }

    /// Reads into the io::Write and returns the next log entry if in-bounds.
    pub async fn read_to_next<W: Write + Unpin>(
        self,
        write: &mut W,
    ) -> Result<Option<LogEntry<'l>>, LogError> {
        let LogEntry { log, index } = self;
        let len = log.file.read_u64().await?;

        let mut hasher = crc32fast::Hasher::new();

        {
            let mut bytes_left: usize = len
                .try_into()
                .expect("Log entry is too large to read on a 32 bit platform.");
            let mut buf = [0; 8 * 1024];

            while bytes_left > 0 {
                let read = bytes_left.min(buf.len());
                let read = log.file.read(&mut buf[..read]).await?;

                hasher.update(&buf[..read]);
                write.write_all(&buf[..read])?;

                bytes_left -= read;
            }
        }

        let checksum = log.file.read_u32().await?;

        if checksum != hasher.finalize() {
            return Err(LogError::BadChecksum);
        }

        let next_index = index + 1;

        if log.first_index + log.len > next_index {
            Ok(Some(LogEntry {
                log,
                index: next_index,
            }))
        } else {
            Ok(None)
        }
    }

    /// Seek forwards to the index. Only forwards traversal is allowed.
    pub async fn seek(self, to_index: u64) -> Result<LogEntry<'l>, LogError> {
        let LogEntry { log, index } = self;

        if to_index > log.first_index + log.len || to_index < index {
            return Err(LogError::OutOfBounds);
        }

        for _ in index..to_index {
            let len = log.file.read_u64().await?;

            // Move forwards through the length of the current log entry and the 4 byte checksum
            log.file
                .seek(SeekFrom::Current((len + 4).try_into().unwrap()))
                .await?;
        }

        Ok(LogEntry {
            log,
            index: to_index,
        })
    }
}

use core::pin::Pin;
use core::task::{Context, Poll};
use futures::future::BoxFuture;
use std::future::Future;
#[pin_project]
pub struct LogIterator<'l> {
    next: Option<LogEntry<'l>>,
    last_index: u64,
    #[pin]
    future: Option<
        Box<
            (dyn Future<Output = Option<Result<(Option<LogEntry<'l>>, Vec<u8>), LogError>>>
                 + Send
                 + 'l),
        >,
    >,
}

impl<'l> Stream for LogIterator<'l> {
    type Item = Result<Vec<u8>, LogError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        if this.future.is_none() {
            let self_last_index = this.last_index;
            let entry = this.next.take();
            let f = async move {
                if entry.is_none() {
                    return None;
                }
                let entry = entry.unwrap();

                if entry.index > *self_last_index {
                    return None;
                };

                let mut content = Vec::new();
                let read_res = entry.read_to_next(&mut content).await;
                match read_res {
                    Ok(next) => Some(Ok((next, content))),
                    Err(err) => Some(Err(err)),
                }
            };
            let new_future = Some(Box::new(f));
            this.future = Pin::new(&mut new_future);
        }

        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn it_works() {
        let path = std::path::Path::new("./wal-log-test");

        let _ = std::fs::remove_file(path);

        let entries = &[b"test".to_vec(), b"foobar".to_vec()];

        {
            let mut log = LogFile::open(path).await.unwrap();

            // write to log
            for entry in entries {
                log.write(&mut entry.clone()).await.unwrap();
            }

            log.flush().await.unwrap();

            // read back and ensure entries match what was written
            for (read, written) in log.iter(..).zip(entries.iter()) {
                assert_eq!(&read.unwrap(), written);
            }
        }

        {
            // test after closing and reopening
            let mut log = LogFile::open(path).await.unwrap();

            let read = log.iter(..).await.unwrap().map(|entry| entry.unwrap());

            assert!(read.eq(entries.to_vec()));
        }

        {
            let mut log = LogFile::open(path).await.unwrap();

            let entry = log.seek(1).await.unwrap();
            let mut content = vec![];
            let next = entry.read_to_next(&mut content).await.unwrap();

            assert_eq!(content, entries[1]);
            assert!(next.is_none());
        }

        {
            let mut log = LogFile::open(path).await.unwrap();

            let entry = log.seek(1).await.unwrap();

            entry.seek(0).await.err().expect("Cannot seek backwards");
        }

        std::fs::remove_file(path).unwrap();
    }

    //     #[test]
    //     fn compaction() {
    //         let path = std::path::Path::new("./wal-log-compaction");

    //         let _ = std::fs::remove_file(path);

    //         let entries = &[
    //             b"test".to_vec(),
    //             b"foobar".to_vec(),
    //             b"bbb".to_vec(),
    //             b"aaaaa".to_vec(),
    //             b"11".to_vec(),
    //             b"222".to_vec(),
    //             [9; 200].to_vec(),
    //             b"bar".to_vec(),
    //         ];

    //         {
    //             let mut log = LogFile::open(path).unwrap();

    //             // write to log
    //             for entry in entries {
    //                 log.write(&mut entry.clone()).unwrap();
    //             }

    //             assert_eq!(log.first_index(), 0);

    //             log.compact(4).unwrap();

    //             assert_eq!(log.first_index(), 4);
    //             assert!(log
    //                 .iter(..)
    //                 .unwrap()
    //                 .map(|a| a.unwrap())
    //                 .eq(entries[4..].to_vec().into_iter()));

    //             log.flush().unwrap();
    //         }

    //         {
    //             let mut log = LogFile::open(path).unwrap();
    //             assert_eq!(log.first_index(), 4);
    //             assert!(log
    //                 .iter(..)
    //                 .unwrap()
    //                 .map(|a| a.unwrap())
    //                 .eq(entries[4..].to_vec().into_iter()));
    //         }

    //         std::fs::remove_file(path).unwrap();
    //     }

    //     #[test]
    //     fn restart() {
    //         let path = std::path::Path::new("./wal-log-restart");

    //         let _ = std::fs::remove_file(path);

    //         let entries = &[
    //             b"test".to_vec(),
    //             b"foobar".to_vec(),
    //             b"bbb".to_vec(),
    //             b"aaaaa".to_vec(),
    //             b"11".to_vec(),
    //             b"222".to_vec(),
    //             [9; 200].to_vec(),
    //             b"bar".to_vec(),
    //         ];

    //         {
    //             let mut log = LogFile::open(path).unwrap();

    //             // write to log
    //             for entry in entries {
    //                 log.write(&mut entry.clone()).unwrap();
    //             }

    //             assert_eq!(log.first_index(), 0);

    //             log.flush().unwrap();
    //         }

    //         {
    //             let mut log = LogFile::open(path).unwrap();
    //             log.restart(3).unwrap();
    //             assert_eq!(log.first_index(), 3);
    //             assert_eq!(log.iter(..).unwrap().collect::<Vec<_>>().len(), 0);
    //         }

    //         {
    //             let mut log = LogFile::open(path).unwrap();
    //             assert_eq!(log.first_index(), 3);
    //             assert_eq!(log.iter(..).unwrap().collect::<Vec<_>>().len(), 0);
    //         }

    //         std::fs::remove_file(path).unwrap();
    //     }

    //     #[test]
    //     fn handles_trimmed_wal() {
    //         let path = std::path::Path::new("./wal-log-test-trimmed");

    //         let _ = std::fs::remove_file(path);

    //         let entries = &[b"test".to_vec(), b"foobar".to_vec()];

    //         {
    //             let mut log = LogFile::open(path).unwrap();

    //             // write to log
    //             for entry in entries {
    //                 log.write(&mut entry.clone()).unwrap();
    //             }

    //             log.flush().unwrap();
    //         }

    //         {
    //             // trim last log entry to cause chaos
    //             let mut file = std::fs::OpenOptions::new()
    //                 .write(true)
    //                 .read(true)
    //                 .open(path)
    //                 .unwrap();
    //             file.set_len(38).unwrap();
    //             file.flush().unwrap();
    //         }

    //         {
    //             // test after closing and reopening
    //             let mut log = LogFile::open(path).unwrap();

    //             let read = log.iter(..).unwrap().map(|entry| entry.unwrap());

    //             assert!(read.eq(entries[..1].to_vec()));
    //         }

    //         std::fs::remove_file(path).unwrap();
    //     }

    //     #[test]
    //     fn last_index_on_empty() {
    //         let path = std::path::Path::new("./wal-log-test-last-index");

    //         {
    //             let log = LogFile::open(path).unwrap();
    //             assert_eq!(log.last_index(), 0);
    //         }

    //         std::fs::remove_file(path).unwrap();
    //     }
}
