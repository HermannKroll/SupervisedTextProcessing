def calc_precision(true_positives, false_positives):
    """
    Computes the precision as the fraction of relevant instances among the retrieved instances.
    """
    if true_positives:
        return true_positives / (true_positives + false_positives)
    else:
        return 0.0


def calc_recall(true_positives, false_negatives):
    """
    Computes the recall as the fraction of relevant instances that were retrieved.
    """
    if true_positives:
        return true_positives / (true_positives + false_negatives)
    else:
        return 0.0


def calc_f1_score(precision: float, recall: float):
    """
    Computes the F1 measure as the harmonic mean of precision and recall.
    """
    if precision or recall:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0.0


def calc_accuracy(true_positives: int, true_negatives: int, false_negatives: int, false_positives: int):
    """
    Computes the accuracy.
    """
    if true_positives or true_negatives:
        return ((true_positives + true_negatives) /
                sum((true_positives, true_negatives, false_negatives, false_positives)))
    else:
        return 0.0


def print_full_metrics(tp: int, fp: int, fn: int, tn: int, na=None, time=None, num_votes=None, desc: str = ""):
    p = calc_precision(tp, fp)
    r = calc_recall(tp, fn)
    f = calc_f1_score(p, r)
    a = calc_accuracy(tp, tn, fn, fp)

    text = list()
    if desc:
        text.append(desc)
    text.append("TP: " + str(tp))
    text.append("FP: " + str(fp))
    text.append("FN: " + str(fn))
    text.append("TN: " + str(tn))
    if na:
        text.append("NA: " + str(na))
    if time:
        text.append("Average time: " + str(time) + "s")
    if num_votes:
        text.append("Total votes : " + str(num_votes))
    text.append("Precision: " + str(round(p, 2)))
    text.append("Recall   : " + str(round(r, 2)))
    text.append("F1       : " + str(round(f, 2)))
    text.append("Accuracy : " + str(round(a, 2)))
    print("\n".join(text), "\n")


def results_to_latex(results: list, desc=""):
    times = [0.0] * 4
    sentences = [0] * 4
    rows = [[], [], [], []]

    def extract_row(tp: int, fp: int, tn: int, fn: int, na: int, **_):
        return [tp, fp, tn, fn, na]

    def extract_row_metrics(**kwargs):
        tp, fp, tn, fn, na = extract_row(**kwargs)
        precision = calc_precision(tp, fp)
        recall = calc_recall(tp, fn)
        f1 = calc_f1_score(precision, recall)

        return round(precision, 2), round(recall, 2), round(f1, 2), na

    for r in results:
        if r:
            times[0] += r["first_only"]["time"]
            times[1] += r["one_yes"]["time"]
            times[2] += r["two_yes"]["time"]
            times[3] += r["three_yes"]["time"]

            sentences[0] += sum(v for k, v in r["first_only"]["data"].items() if k != "na")
            sentences[1] += sum(v for k, v in r["one_yes"]["data"].items() if k != "na")
            sentences[2] += sum(v for k, v in r["two_yes"]["data"].items() if k != "na")
            sentences[3] += sum(v for k, v in r["three_yes"]["data"].items() if k != "na")

    for i in range(len(times)):
        if sentences[i]:
            times[i] /= sentences[i]
        else:
            times[i] = 0.0
        rows[i].append(str(round(times[i], 2)) + "s")

    for r in results:
        if r:
            row_first_only = extract_row_metrics(**r["first_only"]["data"])
            row_one_yes = extract_row_metrics(**r["one_yes"]["data"])
            row_two_yes = extract_row_metrics(**r["two_yes"]["data"])
            row_three_yes = extract_row_metrics(**r["three_yes"]["data"])
        else:
            row_first_only = [None] * 4
            row_one_yes = [None] * 4
            row_two_yes = [None] * 4
            row_three_yes = [None] * 4

        rows[0].extend(row_first_only)
        rows[1].extend(row_one_yes)
        rows[2].extend(row_two_yes)
        rows[3].extend(row_three_yes)

    print("LATEX TABLE ELEMENTS FOR", desc)
    for row in rows:
        print(" & ".join((f"{float(i):.2f}" if isinstance(i, float) else f"{i:>4}") if i else "   -" for i in row), r"\\")
    print()
