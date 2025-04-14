
import os
import pickle
import threading

class ListRecordLoader:
    OFFSET_LENGTH = 8
    def __init__(self, load_path):
        self._sync_lock = threading.Lock()
        self._size = os.path.getsize(load_path)
        self._load_path = load_path
        self._open_file()
        self._scan_file()

    def _open_file(self):
        self._pid = os.getpid()
        self._cache_file = open(self._load_path, 'rb')

    def _check_reopen(self):
        if (self._pid != os.getpid()):
            self._open_file()

    def _scan_file(self):
        record_pos_list = list()
        pos = 0
        while True:
            if pos >= self._size:
                break
            self._cache_file.seek(pos)
            offset = int().from_bytes(
                self._cache_file.read(self.OFFSET_LENGTH),
                byteorder='big', signed=False
            )
            offset = pos + offset
            self._cache_file.seek(offset)

            byte_size = int().from_bytes(
                self._cache_file.read(self.OFFSET_LENGTH),
                byteorder='big', signed=False
            )
            record_pos_list_bytes = self._cache_file.read(byte_size)
            sub_record_pos_list = pickle.loads(record_pos_list_bytes)
            assert isinstance(sub_record_pos_list, list)
            sub_record_pos_list = [[item[0]+pos, item[1]] for item in sub_record_pos_list]
            record_pos_list.extend(sub_record_pos_list)
            pos = self._cache_file.tell()

        self._record_pos_list = record_pos_list

    def get_record(self, idx):
        self._check_reopen()
        record_bytes = self.get_record_bytes(idx)
        record = pickle.loads(record_bytes)
        return record

    def get_record_bytes(self, idx):
        offset, length = self._record_pos_list[idx]
        self._sync_lock.acquire()
        try:
            self._cache_file.seek(offset)
            record_bytes = self._cache_file.read(length)
        finally:
            self._sync_lock.release()
        return record_bytes

    def __len__(self):
        return len(self._record_pos_list)
