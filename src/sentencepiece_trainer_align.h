#ifndef SENTENCEPIECE_TRAINER_ALIGN_H_
#define SENTENCEPIECE_TRAINER_ALGIN_H_

#include <string>
#include <unordered_map>

#include "sentencepiece_processor.h"
#include "unigram_model_trainer.h"
#include "unigram_model.h"

namespace sentencepiece {
class SentencePieceAlignTrainer {
 public:
  static util::Status Train();
  static util::Status Train(const TrainerSpec &trainer_spec_src, const TrainerSpec &trainer_spec_tgt,const NormalizerSpec &normalizer_speck, const NormalizerSpec &denormalizer_spec);

  //joint train
  static util::Status TrainAlign(
        const TrainerSpec &trainer_spec_src,
        const TrainerSpec &trainer_spec_tgt,
        const NormalizerSpec &normalizer_spec,
        const std::unique_ptr<unigram::Trainer> &trainer_src,
        const std::unique_ptr<unigram::Trainer> &trainer_tgt
        );


    static util::Status PruneSentencePiecesJoint();
    static util::Status PruneSentencePiecesJoint(
            const std::unique_ptr<unigram::Trainer> &trainer_src,
            const std::unique_ptr<unigram::Trainer> &trainer_tgt
            );

    static util::Status PruneSentencePiecesJoint(
            const std::unique_ptr<unigram::Trainer> &trainer_src,
            const std::unique_ptr<unigram::Trainer> &trainer_tgt,
            const unigram::TrainerModel &model_src,
            const unigram::TrainerModel &model_tgt
            );
 private:
  SentencePieceAlignTrainer() {}
  ~SentencePieceAlignTrainer() {}
};

}  // namespace sentencepiece

#endif
