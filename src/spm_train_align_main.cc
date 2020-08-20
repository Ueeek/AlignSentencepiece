#include <map>

#include "init.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_trainer_align.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/strings/ascii.h"
#include "third_party/absl/strings/str_split.h"
#include "util.h"


using sentencepiece::NormalizerSpec;
using sentencepiece::TrainerSpec;

namespace {
    static sentencepiece::TrainerSpec kDefaultTrainerSpec;
    static sentencepiece::NormalizerSpec kDefaultNormalizerSpec;
}

//ABSL_FLAG(std::string, input, "", "comma separated list of input sentences");
ABSL_FLAG(std::string, input, "", "comma separated list of input sentences");
ABSL_FLAG(std::string, input_format, kDefaultTrainerSpec.input_format(),
          "Input format. Supported format is `text` or `tsv`.");
ABSL_FLAG(std::string, model_prefix, "", "output model prefix");
//ABSL_FLAG(std::string, model_prefix, "", "output model prefix");
ABSL_FLAG(std::string, model_type, "unigram",
          "model algorithm: unigram, bpe, word or char");
ABSL_FLAG(int32, vocab_size, kDefaultTrainerSpec.vocab_size(),
          "vocabulary size");
ABSL_FLAG(std::string, accept_language, "",
          "comma-separated list of languages this model can accept");
ABSL_FLAG(int32, self_test_sample_size,
          kDefaultTrainerSpec.self_test_sample_size(),
          "the size of self test samples");
ABSL_FLAG(double, character_coverage, kDefaultTrainerSpec.character_coverage(),
          "character coverage to determine the minimum symbols");
ABSL_FLAG(int32, input_sentence_size, kDefaultTrainerSpec.input_sentence_size(),
          "maximum size of sentences the trainer loads");
ABSL_FLAG(bool, shuffle_input_sentence,
          kDefaultTrainerSpec.shuffle_input_sentence(),
          "Randomly sample input sentences in advance. Valid when "
          "--input_sentence_size > 0");
ABSL_FLAG(int32, seed_sentencepiece_size,
          kDefaultTrainerSpec.seed_sentencepiece_size(),
          "the size of seed sentencepieces");
ABSL_FLAG(double, shrinking_factor, kDefaultTrainerSpec.shrinking_factor(),
          "Keeps top shrinking_factor pieces with respect to the loss");
ABSL_FLAG(int32, num_threads, kDefaultTrainerSpec.num_threads(),
          "number of threads for training");
ABSL_FLAG(int32, num_sub_iterations, kDefaultTrainerSpec.num_sub_iterations(),
          "number of EM sub-iterations");
ABSL_FLAG(int32, max_sentencepiece_length,
          kDefaultTrainerSpec.max_sentencepiece_length(),
          "maximum length of sentence piece");
ABSL_FLAG(int32, max_sentence_length, kDefaultTrainerSpec.max_sentence_length(),
          "maximum length of sentence in byte");
ABSL_FLAG(bool, split_by_unicode_script,
          kDefaultTrainerSpec.split_by_unicode_script(),
          "use Unicode script to split sentence pieces");
ABSL_FLAG(bool, split_by_number, kDefaultTrainerSpec.split_by_number(),
          "split tokens by numbers (0-9)");
ABSL_FLAG(bool, split_by_whitespace, kDefaultTrainerSpec.split_by_whitespace(),
          "use a white space to split sentence pieces");
ABSL_FLAG(bool, split_digits, kDefaultTrainerSpec.split_digits(),
          "split all digits (0-9) into separate pieces");
ABSL_FLAG(bool, treat_whitespace_as_suffix,
          kDefaultTrainerSpec.treat_whitespace_as_suffix(),
          "treat whitespace marker as suffix instead of prefix.");
ABSL_FLAG(std::string, control_symbols, "",
          "comma separated list of control symbols");
ABSL_FLAG(std::string, user_defined_symbols, "",
          "comma separated list of user defined symbols");
ABSL_FLAG(std::string, required_chars, "",
          "UTF8 characters in this flag are always used in the character "
          "set regardless of --character_coverage");
ABSL_FLAG(bool, byte_fallback, kDefaultTrainerSpec.byte_fallback(),
          "decompose unknown pieces into UTF-8 byte pieces");
ABSL_FLAG(bool, vocabulary_output_piece_score,
          kDefaultTrainerSpec.vocabulary_output_piece_score(),
          "Define score in vocab file");
ABSL_FLAG(std::string, normalization_rule_name, "nmt_nfkc",
          "Normalization rule name. "
          "Choose from nfkc or identity");
ABSL_FLAG(std::string, normalization_rule_tsv, "",
          "Normalization rule TSV file. ");
ABSL_FLAG(std::string, denormalization_rule_tsv, "",
          "Denormalization rule TSV file.");
ABSL_FLAG(bool, add_dummy_prefix, kDefaultNormalizerSpec.add_dummy_prefix(),
          "Add dummy whitespace at the beginning of text");
ABSL_FLAG(bool, remove_extra_whitespaces,
          kDefaultNormalizerSpec.remove_extra_whitespaces(),
          "Removes leading, trailing, and "
          "duplicate internal whitespace");
ABSL_FLAG(bool, hard_vocab_limit, kDefaultTrainerSpec.hard_vocab_limit(),
          "If set to false, --vocab_size is considered as a soft limit.");
ABSL_FLAG(bool, use_all_vocab, kDefaultTrainerSpec.use_all_vocab(),
          "If set to true, use all tokens as vocab. "
          "Valid for word/char models.");
ABSL_FLAG(int32, unk_id, kDefaultTrainerSpec.unk_id(),
          "Override UNK (<unk>) id.");
ABSL_FLAG(int32, bos_id, kDefaultTrainerSpec.bos_id(),
          "Override BOS (<s>) id. Set -1 to disable BOS.");
ABSL_FLAG(int32, eos_id, kDefaultTrainerSpec.eos_id(),
          "Override EOS (</s>) id. Set -1 to disable EOS.");
ABSL_FLAG(int32, pad_id, kDefaultTrainerSpec.pad_id(),
          "Override PAD (<pad>) id. Set -1 to disable PAD.");
ABSL_FLAG(std::string, unk_piece, kDefaultTrainerSpec.unk_piece(),
          "Override UNK (<unk>) piece.");
ABSL_FLAG(std::string, bos_piece, kDefaultTrainerSpec.bos_piece(),
          "Override BOS (<s>) piece.");
ABSL_FLAG(std::string, eos_piece, kDefaultTrainerSpec.eos_piece(),
          "Override EOS (</s>) piece.");
ABSL_FLAG(std::string, pad_piece, kDefaultTrainerSpec.pad_piece(),
          "Override PAD (<pad>) piece.");
ABSL_FLAG(std::string, unk_surface, kDefaultTrainerSpec.unk_surface(),
          "Dummy surface string for <unk>. In decoding <unk> is decoded to "
          "`unk_surface`.");
ABSL_FLAG(bool, train_extremely_large_corpus,
          kDefaultTrainerSpec.train_extremely_large_corpus(),
          "Increase bit depth for unigram tokenization.");

int main(int argc, char *argv[]) {
  sentencepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);

  LOG(INFO)<<"train_align::main() called";
  CHECK(!absl::GetFlag(FLAGS_input).empty());
  CHECK(!absl::GetFlag(FLAGS_model_prefix).empty());

  sentencepiece::TrainerSpec trainer_spec_src, trainer_spec_tgt;
  sentencepiece::NormalizerSpec normalizer_spec;
  NormalizerSpec denormalizer_spec;
  
// Populates the value from flags to spec.
#define SetTrainerSpecFromFlagSrc(name) \
  trainer_spec_src.set_##name(absl::GetFlag(FLAGS_##name));
#define SetTrainerSpecFromFlagTgt(name) \
  trainer_spec_tgt.set_##name(absl::GetFlag(FLAGS_##name));

#define SetNormalizerSpecFromFlag(name) \
  normalizer_spec.set_##name(absl::GetFlag(FLAGS_##name));

#define SetRepeatedTrainerSpecFromFlagSrc(name)                                \
  if (!absl::GetFlag(FLAGS_##name).empty()) {                               \
    for (const auto &v :                                                    \
         sentencepiece::util::StrSplitAsCSV(absl::GetFlag(FLAGS_##name))) { \
      trainer_spec_src.add_##name(v);                                           \
    }                                                                       \
  }
#define SetRepeatedTrainerSpecFromFlagTgt(name)                                \
  if (!absl::GetFlag(FLAGS_##name).empty()) {                               \
    for (const auto &v :                                                    \
         sentencepiece::util::StrSplitAsCSV(absl::GetFlag(FLAGS_##name))) { \
      trainer_spec_tgt.add_##name(v);                                           \
    }                                                                       \
  }
  SetRepeatedTrainerSpecFromFlagSrc(input);
  SetTrainerSpecFromFlagSrc(input_format);
  SetTrainerSpecFromFlagSrc(model_prefix);
  SetTrainerSpecFromFlagSrc(vocab_size);
  SetTrainerSpecFromFlagSrc(self_test_sample_size);
  SetTrainerSpecFromFlagSrc(character_coverage);
  SetTrainerSpecFromFlagSrc(input_sentence_size);
  SetTrainerSpecFromFlagSrc(shuffle_input_sentence);
  SetTrainerSpecFromFlagSrc(seed_sentencepiece_size);
  SetTrainerSpecFromFlagSrc(shrinking_factor);
  SetTrainerSpecFromFlagSrc(num_threads);
  SetTrainerSpecFromFlagSrc(num_sub_iterations);
  SetTrainerSpecFromFlagSrc(max_sentencepiece_length);
  SetTrainerSpecFromFlagSrc(max_sentence_length);
  SetTrainerSpecFromFlagSrc(split_by_unicode_script);
  SetTrainerSpecFromFlagSrc(split_by_whitespace);
  SetTrainerSpecFromFlagSrc(split_by_number);
  SetTrainerSpecFromFlagSrc(split_digits);
  SetTrainerSpecFromFlagSrc(byte_fallback);
  SetTrainerSpecFromFlagSrc(treat_whitespace_as_suffix);
  SetTrainerSpecFromFlagSrc(hard_vocab_limit);
  SetTrainerSpecFromFlagSrc(use_all_vocab);
  SetTrainerSpecFromFlagSrc(unk_id);
  SetTrainerSpecFromFlagSrc(bos_id);
  SetTrainerSpecFromFlagSrc(eos_id);
  SetTrainerSpecFromFlagSrc(pad_id);
  SetTrainerSpecFromFlagSrc(unk_piece);
  SetTrainerSpecFromFlagSrc(bos_piece);
  SetTrainerSpecFromFlagSrc(eos_piece);
  SetTrainerSpecFromFlagSrc(pad_piece);
  SetTrainerSpecFromFlagSrc(unk_surface);
  SetTrainerSpecFromFlagSrc(required_chars);
  SetTrainerSpecFromFlagSrc(vocabulary_output_piece_score);
  SetRepeatedTrainerSpecFromFlagSrc(accept_language);
  SetRepeatedTrainerSpecFromFlagSrc(control_symbols);
  SetRepeatedTrainerSpecFromFlagSrc(user_defined_symbols);
  SetTrainerSpecFromFlagSrc(train_extremely_large_corpus);

  SetRepeatedTrainerSpecFromFlagTgt(input);
  SetTrainerSpecFromFlagTgt(input_format);
  SetTrainerSpecFromFlagTgt(model_prefix);
  SetTrainerSpecFromFlagTgt(vocab_size);
  SetTrainerSpecFromFlagTgt(self_test_sample_size);
  SetTrainerSpecFromFlagTgt(character_coverage);
  SetTrainerSpecFromFlagTgt(input_sentence_size);
  SetTrainerSpecFromFlagTgt(shuffle_input_sentence);
  SetTrainerSpecFromFlagTgt(seed_sentencepiece_size);
  SetTrainerSpecFromFlagTgt(shrinking_factor);
  SetTrainerSpecFromFlagTgt(num_threads);
  SetTrainerSpecFromFlagTgt(num_sub_iterations);
  SetTrainerSpecFromFlagTgt(max_sentencepiece_length);
  SetTrainerSpecFromFlagTgt(max_sentence_length);
  SetTrainerSpecFromFlagTgt(split_by_unicode_script);
  SetTrainerSpecFromFlagTgt(split_by_whitespace);
  SetTrainerSpecFromFlagTgt(split_by_number);
  SetTrainerSpecFromFlagTgt(split_digits);
  SetTrainerSpecFromFlagTgt(byte_fallback);
  SetTrainerSpecFromFlagTgt(treat_whitespace_as_suffix);
  SetTrainerSpecFromFlagTgt(hard_vocab_limit);
  SetTrainerSpecFromFlagTgt(use_all_vocab);
  SetTrainerSpecFromFlagTgt(unk_id);
  SetTrainerSpecFromFlagTgt(bos_id);
  SetTrainerSpecFromFlagTgt(eos_id);
  SetTrainerSpecFromFlagTgt(pad_id);
  SetTrainerSpecFromFlagTgt(unk_piece);
  SetTrainerSpecFromFlagTgt(bos_piece);
  SetTrainerSpecFromFlagTgt(eos_piece);
  SetTrainerSpecFromFlagTgt(pad_piece);
  SetTrainerSpecFromFlagTgt(unk_surface);
  SetTrainerSpecFromFlagTgt(required_chars);
  SetTrainerSpecFromFlagTgt(vocabulary_output_piece_score);
  SetRepeatedTrainerSpecFromFlagTgt(accept_language);
  SetRepeatedTrainerSpecFromFlagTgt(control_symbols);
  SetRepeatedTrainerSpecFromFlagTgt(user_defined_symbols);
  SetTrainerSpecFromFlagTgt(train_extremely_large_corpus);

  normalizer_spec.set_name(absl::GetFlag(FLAGS_normalization_rule_name));
  SetNormalizerSpecFromFlag(normalization_rule_tsv);
  SetNormalizerSpecFromFlag(add_dummy_prefix);
  SetNormalizerSpecFromFlag(remove_extra_whitespaces);


  CHECK_OK(sentencepiece::SentencePieceAlignTrainer::Train());
  CHECK_OK(sentencepiece::SentencePieceAlignTrainer::Train(trainer_spec_src,trainer_spec_tgt,normalizer_spec, denormalizer_spec));

  return 0;
}
