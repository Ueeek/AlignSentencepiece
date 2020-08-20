
#include <string>
#include <vector>

#include "builder.h"
#include "common.h"
#include "normalizer.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_trainer_align.h"
#include "sentencepiece_trainer.h"
#include "spec_parser_align.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/strings/strip.h"
#include "util.h"

#include "unigram_model.h"
#include "unigram_model_trainer.h"


//TODO
//とりあえず。unigramMOdelのみに対して。実装する。
//必要なら。あとでmodel factoryして。

namespace sentencepiece {
namespace {
static constexpr char kDefaultNormalizerName[] = "nmt_nfkc";
}  // namespace

// static
util::Status SentencePieceAlignTrainer::Train() {
  LOG(INFO)<<"align trainer called";
  return util::OkStatus();
}

util::Status SentencePieceAlignTrainer::Train(
        const TrainerSpec &trainer_spec_src,
        const TrainerSpec &trainer_spec_tgt,
        const NormalizerSpec &normalizer_spec,
        const NormalizerSpec &denormalizer_spec) {

    //SentencePieceTrainer::Train(trainer_spec_src,normalizer_spec,denormalizer_spec);
    //SentencePieceTrainer::Train(trainer_spec_tgt,normalizer_spec,denormalizer_spec);
    auto copied_normalizer_spec = normalizer_spec;
    auto copied_denormalizer_spec = denormalizer_spec;
    //auto trainer_src = TrainerFactory::Create(trainer_spec_src,copied_normalizer_spec,copied_denormalizer_spec);
    //auto trainer_tgt = TrainerFactory::Create(trainer_spec_src,copied_normalizer_spec,copied_denormalizer_spec);
    std::unique_ptr<unigram::Trainer> trainer_src = absl::make_unique<unigram::Trainer>(trainer_spec_src,copied_normalizer_spec,copied_denormalizer_spec);
    std::unique_ptr<unigram::Trainer> trainer_tgt = absl::make_unique<unigram::Trainer>(trainer_spec_tgt,copied_normalizer_spec,copied_denormalizer_spec);

    std::string info = absl::StrCat(
            PrintProto(trainer_spec_src,"trainer_spec_src"),
            PrintProto(trainer_spec_tgt,"trainer_spec_tgt"),
            PrintProto(copied_normalizer_spec,"nomalizer_spec"));

    LOG(INFO)<<"Starts training with :\n" << info;

    TrainAlign(trainer_spec_src, trainer_spec_tgt, normalizer_spec,trainer_src,trainer_tgt);
                                        
  return util::OkStatus();
}

util::Status SentencePieceAlignTrainer::TrainAlign(
        const TrainerSpec &trainer_spec_src,
        const TrainerSpec &trainer_spec_tgt,
        const NormalizerSpec &normalizer_spec,
        const std::unique_ptr<unigram::Trainer> &trainer_src,
        const std::unique_ptr<unigram::Trainer> &trainer_tgt
        ){



    LOG(INFO)<<"starts joint train";

    CHECK_EQ_OR_RETURN(TrainerSpec::UNIGRAM, trainer_spec_src.model_type());
    CHECK_EQ_OR_RETURN(TrainerSpec::UNIGRAM, trainer_spec_tgt.model_type());
    CHECK_OR_RETURN(normalizer_spec.escape_whitespaces());

    sentencepiece::unigram::TrainerModel model_src(trainer_spec_src, normalizer_spec);
    sentencepiece::unigram::TrainerModel model_tgt(trainer_spec_src, normalizer_spec);

    RETURN_IF_ERROR(model_src.status());
    RETURN_IF_ERROR(model_tgt.status());

    RETURN_IF_ERROR(trainer_src->LoadSentences());
    RETURN_IF_ERROR(trainer_tgt->LoadSentences());


    if(trainer_spec_src.train_extremely_large_corpus() || trainer_spec_tgt.train_extremely_large_corpus()){
      model_src.SetSentencePieces(trainer_src->MakeSeedSentencePieces<int64>());
      model_tgt.SetSentencePieces(trainer_tgt->MakeSeedSentencePieces<int64>());
     } else{
      model_src.SetSentencePieces(trainer_src->MakeSeedSentencePieces<int32>());
      model_tgt.SetSentencePieces(trainer_tgt->MakeSeedSentencePieces<int32>());
     }


     if (trainer_spec_src.split_by_whitespace()){
        trainer_src->SplitSentencesByWhitespace();
     }
     if (trainer_spec_tgt.split_by_whitespace()){
         trainer_tgt->SplitSentencesByWhitespace();
     }

     LOG(INFO)<<"SRC:::Using "<< trainer_src->sentences_.size() << "sentences for EM Training";
     LOG(INFO)<<"TGT:::Using "<< trainer_tgt->sentences_.size() << "sentences for EM Training";

     trainer_src->desired_vocab_size_ = static_cast<size_t>(trainer_spec_src.vocab_size()*1.1);
     trainer_tgt->desired_vocab_size_ = static_cast<size_t>(trainer_spec_tgt.vocab_size()*1.1);

    trainer_src->Train();
    trainer_tgt->Train();




    return util::OkStatus();
}
}  // namespace sentencepiece
