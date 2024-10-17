from stp.document import parse_ddi_run, parse_chemprot_run, parse_cdr_run, Document


def analyze_documents(parse_function, run_name: str) -> (int, int, int):
    documents = parse_function(run_name=run_name)

    relevant_sentences = 0
    words_per_sentence = 0

    doc: Document
    for doc in documents:
        for s in doc.get_sentences():
            words_per_sentence += len(s.text.split())
            relevant_sentences += 1

    print()
    print(run_name)
    print("Documents           :", len(documents))
    print("Relevant Sentences  :", relevant_sentences)
    if relevant_sentences == 0:
        print("No relevant sentences --> excluded from evaluation")
        return 0, 0, 0
    print("Mean Tokens/Sentence:", words_per_sentence / relevant_sentences)

    return len(documents), relevant_sentences, words_per_sentence


def main():
    runs = {
        parse_cdr_run: [
            # "CDR_sample.txt",
            "CDR_TrainingSet.PubTator.txt",
            "CDR_DevelopmentSet.PubTator.txt",
            "CDR_TestSet.PubTator.txt",
        ],
        parse_chemprot_run: [
            # "sample",
            "training",
            "development",
            "test",
        ],
        parse_ddi_run: [
            "Train",
            "Test",
        ]
    }

    num_docs = 0
    relevant_sentences = 0
    words_per_sentence = 0

    for function, run_names in runs.items():
        for run_name in run_names:
            run_num_docs, run_relevant_sentences, run_words_per_sentence = analyze_documents(function, run_name)
            num_docs += run_num_docs
            relevant_sentences += run_relevant_sentences
            words_per_sentence += run_words_per_sentence

    print()
    print("Overall Result:")
    print("Documents           :", num_docs)
    print("Relevant Sentences  :", relevant_sentences)
    print("Mean Tokens/Sentence:", round(words_per_sentence / relevant_sentences, 2))


if __name__ == "__main__":
    main()
