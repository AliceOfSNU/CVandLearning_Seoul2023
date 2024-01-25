from defines import LABELS, VOCAB, SOS_TOKEN, EOS_TOKEN
import Levenshtein
import ctcdecode
import torch
from abc import *

class DecodeUtilBase(metaclass=ABCMeta):
    @abstractmethod
    def decode_prediction(self, output, PHONEME_MAP=None):
        pass

# decoder using c++ ctcdecode library. (beam search)
class CTCDecodeUtil(DecodeUtilBase):
    def __init__(self, beam_width):
        super().__init__()
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(
            LABELS,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=beam_width, #this should not exceed 50
            num_processes=4, #this should be ~#of cpus
            blank_id=0,
            log_probs_input=True
        )
        
    def decode_prediction(self, output, output_lens, PHONEME_MAP= LABELS):
        # decode output B*T*V(=VOCAB_SIZE) using beam search
        # CTC decoder
        beam_results, beam_scores, timesteps, output_lens  = self.ctc_decoder.decode(output, seq_lens= output_lens) #lengths - list of lengths
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

# decoder using greedy strategy
def indices_to_chars(indices, vocab):
    tokens = []
    for i in indices: # This loops through all the indices
        if int(i) == SOS_TOKEN: # If SOS is encountered, dont add it to the final list
            continue
        elif int(i) == EOS_TOKEN: # If EOS is encountered, stop the decoding process
            break
        else:
            tokens.append(vocab[i])
    return "".join(tokens)

class GreedyDecodeUtil(DecodeUtilBase):
    def decode_prediction(self, output, output_lens=None, PHONEME_MAP=VOCAB):
        pred_strings = []
        indices = torch.argmax(output, dim=-1)
        for i in range(indices.shape[0]):
            str = indices_to_chars(indices[i].tolist(), PHONEME_MAP)
            pred_strings.append(str)
        return pred_strings
    
#code from course notebook
def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP= LABELS): # y - sequence of integers

    dist            = 0
    batch_size      = label.shape[0]

    pred_strings    = decoder.decode_prediction(output, output_lens, PHONEME_MAP)

    for i in range(batch_size):
        pred_string = pred_strings[i]
        label_string = ""
        for t in range(label_lens[i]):
            label_string += PHONEME_MAP[label[i, t].item()]
        dist += Levenshtein.distance(pred_string, label_string)

    dist /= batch_size
    return dist

