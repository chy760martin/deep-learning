import numpy as np

def postprocess_qa_predictions(examples, features, predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = predictions
    predictions = {}

    for example, feature, start_logit, end_logit in zip(examples, features, all_start_logits, all_end_logits):
        start_indexes = np.argsort(start_logit)[-n_best_size:]
        end_indexes = np.argsort(end_logit)[-n_best_size:]

        valid_answers = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index <= end_index and end_index - start_index + 1 <= max_answer_length:
                    offset = feature["offset_mapping"][start_index:end_index+1]
                    if offset:
                        start_char = offset[0][0]
                        end_char = offset[-1][1]
                        valid_answers.append({
                            "score": start_logit[start_index] + end_logit[end_index],
                            "text": feature["context"][start_char:end_char]
                        })

        if valid_answers:
            best_answer = max(valid_answers, key=lambda x: x["score"])
            predictions[example["id"]] = best_answer["text"]
        else:
            predictions[example["id"]] = ""

    return [{"id": k, "prediction_text": v} for k, v in predictions.items()]