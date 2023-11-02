from LatexUtils import create_regular_table, create_multi_table
import matplotlib.pyplot as plt





def create_summary_is():
    input = open("input.txt")
    data = {}
    for line in input:
        # Check if the line contains dataset and model information
        if "Min:" in line and "Max:" in line and "Q1:" in line and "Q3:" in line and "Median:" in line:
            dataset, model = line.split("Dataset: ")[1].split(" Model: ")[0], line.split("Model: ")[1].split(" ")[0]
            # Create an empty dictionary for the dataset if it does not exist
            if model not in data:
                data[model] = {}
            # Create an empty dictionary for the model if it does not exist
            if dataset not in data[model]:
                data[model][dataset] = {}
            # Extract the data values from the line
            min_val = round(float(line.split("Min: ")[1].split(" ")[0]), 3)
            max_val = round(float(line.split("Max: ")[1].split(" ")[0]), 3)
            q1_val = round(float(line.split("Q1: ")[1].split(" ")[0]), 3)
            q3_val = round(float(line.split("Q3: ")[1].split(" ")[0]), 3)
            median_val = round(float(line.split("Median: ")[1]), 3)
            # Store the data values in the dictionary
            data[model][dataset]["Min"] = min_val
            data[model][dataset]["Max"] = max_val
            data[model][dataset]["Q1"] = q1_val
            data[model][dataset]["Q3"] = q3_val
            data[model][dataset]["Median"] = median_val

    table = create_multi_table(data, "Model name")
    latex = table.to_latex(index_names=False, index=False)
    latex = latex.replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "").replace("\n",
                                                                                                        " \\hline \n")

    latex = latex.replace("{l}", "{c||}").replace("0.", ".").replace("1.0", "1")
    print(latex)


if __name__ == "__main__":
    create_size_predictions_table()
    # create_summary_is()
    # create_box_plots()
    # check_for_rule_type_agreement_hetionet()
