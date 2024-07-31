import sqlite3
import numpy as np
from hash import Hashing

import sqlite3
import numpy as np
from hash import Hashing

class CacheDatabase:
    """
    A class to handle caching of numpy arrays in an SQLite database.
    """

    def __init__(self, db_path="db/cache.db"):
        """
        Initializes the CacheDatabase instance.
        """
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self._create_table()
        self.hash_function = Hashing()

    def _create_table(self):
        """
        Creates the cache table if it does not exist.
        The table has the following columns:
            - key: TEXT PRIMARY KEY
            - value: BLOB
            - shape: TEXT
            - dtype: TEXT
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                shape TEXT,
                dtype TEXT
            )
            """
        )
        self.connection.commit()

    def execute(self, img_path, result_array):
        """
        Inserts a numpy array into the cache database with a hash key.
        """
        hash_value = self._hashing(img_path)
        shape = ','.join(map(str, result_array.shape))
        dtype = str(result_array.dtype)
        blob = result_array.tobytes()
        self.cursor.execute(
            "INSERT INTO cache (key, value, shape, dtype) VALUES (?, ?, ?, ?)", 
            (hash_value, blob, shape, dtype)
        )
        self.connection.commit()

    def find_by_path(self, img_path):
        """
        Retrieves a numpy array from the cache database using an image path.

        Parameters:
        -----------
        img_path : str
            The path to the image file.

        Returns:
        --------
        numpy.ndarray or None
            The cached numpy array if found, otherwise None.
        """
        hash_value = self._hashing(img_path)
        return self.find_by_key(hash_value)

    def find_by_key(self, key):
        """
        Retrieves a numpy array from the cache database using a hash key.

        Parameters:
        -----------
        key : str
            The hash key of the cached numpy array.

        Returns:
        --------
        numpy.ndarray or None
            The cached numpy array if found, otherwise None.
        """
        self.cursor.execute("SELECT value, shape, dtype FROM cache WHERE key = ?", (key,))
        result = self.cursor.fetchone()
        
        if result:
            blob = result[0]
            shape = tuple(map(int, result[1].split(',')))
            dtype = result[2]
            array = np.frombuffer(blob, dtype=dtype).reshape(shape)
            return array
        else:
            print(f"No entry found for key: {key}")
            return None

    def _hashing(self, img_path):
        """
        Generates a hash value for a given image path.

        """
        return self.hash_function.get_hash(img_path)

    def close(self):
        """
        Closes the connection to the database.
        """
        self.connection.close()
