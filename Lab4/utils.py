from defines import LABELS
import Levenshtein

def decode_prediction(output, output_lens, decoder, PHONEME_MAP= LABELS):
    # decode output B*T*V(=VOCAB_SIZE) using beam search
    # CTC decoder
    beam_results, beam_scores, timesteps, output_lens  = decoder.decode(output, seq_lens= output_lens) #lengths - list of lengths
    # beam_scores -> small the better. returns in sorted order
    pred_strings  = []
    # output_lens: B * BW(beam width)
    for i in range(output_lens.shape[0]):
        str = ""
        for t in range(output_lens[i, 0].item()):
            # beam_results : B * BW(beam width) * T
            str += PHONEME_MAP[beam_results[i, 0, t].item()]
        pred_strings.append(str)
        
    return pred_strings

#code from course notebook
def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP= LABELS): # y - sequence of integers

    dist            = 0
    batch_size      = label.shape[0]

    pred_strings    = decode_prediction(output, output_lens, decoder, PHONEME_MAP)

    for i in range(batch_size):
        pred_string = pred_strings[i]
        label_string = ""
        for t in range(label_lens[i]):
            label_string += PHONEME_MAP[label[i, t].item()]
        dist += Levenshtein.distance(pred_string, label_string)

    dist /= batch_size # TODO: Uncomment this, but think about why we are doing this
    return dist