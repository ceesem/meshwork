DEFAULT_VOXEL_RESOLUTION = [4, 4, 40]


class InputError(Exception):
    def __init__(self, message):
        self.message = message


def unique_column_name(base_name, suffix, df):
    col_name = f"{base_name}_{suffix}"
    if col_name in df.columns:
        ii = 0
        while True:
            test_col_name = f"{col_name}_{ii}"
            ii += 1
            if test_col_name not in df.columns:
                col_name = test_col_name
                break
    return col_name
