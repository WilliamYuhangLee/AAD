import pandas as pd


def merge_pareto_files(numbers):

    def read_pareto_file(number):
        prefix = "../../emade/paretoFitnessTitanic"
        suffix = ".txt"
        columns = ["Generation", "FP", "FN", "Number of Elements"]
        file_name = prefix + str(number) + suffix
        return pd.read_csv(file_name, header=None, names=columns)

    result = read_pareto_file(numbers[0])
    size = result["Generation"].max() + 1

    for number in numbers[1:]:
        new = read_pareto_file(number)
        rows = new["Generation"].max() + 1
        new["Generation"] += size
        size += rows
        result = pd.concat([result, new], ignore_index=True)

    return result


result = merge_pareto_files([23404, 30665, 7165, 1502])
print(result)
result.to_csv("pareto_combined.csv")
