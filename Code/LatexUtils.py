import pandas as pd


def create_multi_table(data, outer_key_name):
    headers = []
    subheaders = []
    for outer_key in data:
        for inner_key in data[outer_key]:
            if inner_key not in headers:
                headers.append(inner_key)

            for inner_inner_key in data[outer_key][inner_key]:
                if inner_inner_key not in subheaders:
                    subheaders.append(inner_inner_key)

    columns_list = [[outer_key_name, ""]]
    for val1 in headers:
        for val2 in subheaders:
            columns_list.append([val1, val2])

    column_names = pd.DataFrame(columns_list)
    columns = pd.MultiIndex.from_frame(column_names)
    rows = []
    for outer_key in data:
        to_append = [outer_key]
        for inner_key in data[outer_key]:
            for inner_inner_key in data[outer_key][inner_key]:
                to_append.append(data[outer_key][inner_key][inner_inner_key])

        rows.append(to_append)

    df = pd.DataFrame(rows, columns=columns)
    return df


def create_regular_table(data, outer_key_name):
    headers = []
    for outer_key in data:
        for inner_key in data[outer_key]:
            if inner_key not in headers:
                headers.append(inner_key)

    rows = []
    for outer_key in data:
        to_append = [outer_key]
        for inner_key in data[outer_key]:
            to_append.append(data[outer_key][inner_key])

        rows.append(to_append)

    df = pd.DataFrame(rows, columns=[outer_key_name] + headers)
    return df
