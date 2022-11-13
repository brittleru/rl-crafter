import unittest
from src.utils.string_utils import get_dir_name_as_img


class TestGetDirNameAsImg(unittest.TestCase):
    def test_windows(self):
        expected_windows = "AAIT.png"
        windows_path = "C:\\Study\\Master\\Anul-2\\Sem-2\\AAIT"

        result_windows = get_dir_name_as_img(windows_path)
        self.assertEqual(expected_windows, result_windows,
                         f"Expected \n\t{expected_windows} But got \n\t{result_windows}")

    def test_unix(self):
        expected_unix = "Fun.png"
        unix_path = "~/Desktop/AAIT/Fun"

        result_unix = get_dir_name_as_img(unix_path)
        self.assertEqual(expected_unix, result_unix, f"Expected \n\t{expected_unix} But got \n\t{result_unix}")

    def test_jpg(self):
        file_type = "jpg"
        expected = f"AAIT.{file_type}"
        path = "C:\\Study\\Master\\Anul-2\\Sem-2\\AAIT"

        result = get_dir_name_as_img(path, file_type)
        self.assertEqual(expected, result, f"Expected \n\t{expected} \nBut got \n\t{result}")

    def test_negative(self):
        expected = "AAIT.jpg"
        path = "C:\\Study\\Master\\Anul-2\\Sem-2\\AAIT"

        result = get_dir_name_as_img(path)
        self.assertNotEqual(expected, result, f"Expected \n\t{expected} \nBut got \n\t{result}")
