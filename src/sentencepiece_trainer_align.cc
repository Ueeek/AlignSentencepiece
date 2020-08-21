
#include <string>
#include <vector>

#include<typeinfo>

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

    unigram::TrainerModel model_src(trainer_spec_src, normalizer_spec);
    unigram::TrainerModel model_tgt(trainer_spec_src, normalizer_spec);

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
     //LOG(INFO)<<"type"<<typeid(model_src).name();
     //std::cout<<"type::trainer"<<typeid(trainer_src).name()<<std::endl;
     //std::cout<<"type::model_src"<<typeid(model_src).name()<<std::endl;
     //std::cout<<"type::&model_src"<<typeid(&model_src).name()<<std::endl;

     trainer_src->desired_vocab_size_ = static_cast<size_t>(trainer_spec_src.vocab_size()*1.1);
     trainer_tgt->desired_vocab_size_ = static_cast<size_t>(trainer_spec_tgt.vocab_size()*1.1);

     while(true){
         LOG(INFO)<<"while";
         for(int iter=0;iter<trainer_spec_src.num_sub_iterations();++iter){
             LOG(INFO)<<"inside src EM";
             //EM
             float objective_src  = 0.0;
             int64 num_tokens_src=0;
             const auto expected_src = trainer_src->RunEStep(model_src,&objective_src,&num_tokens_src);
             auto new_sentencepieces = trainer_src->RunMStep(model_src,expected_src);
             model_src.SetSentencePieces(std::move(new_sentencepieces));
             LOG(INFO) << "EM sub_iter=" << iter << " size=" << model_src.GetPieceSize()
                    << " obj=" << objective_src << " num_tokens=" << num_tokens_src
                    << " num_tokens/piece="
                    << 1.0 * num_tokens_src / model_src.GetPieceSize();
         }
         LOG(INFO)<<trainer_spec_src.num_sub_iterations();
         for(int iter=0;iter<trainer_spec_tgt.num_sub_iterations();++iter){
             //EM
             float objective_tgt  = 0.0;
             int64 num_tokens_tgt=0;
             const auto expected_tgt = trainer_tgt->RunEStep(model_tgt,&objective_tgt,&num_tokens_tgt);
             auto new_sentencepieces = trainer_tgt->RunMStep(model_tgt,expected_tgt);
             model_tgt.SetSentencePieces(std::move(new_sentencepieces));
             LOG(INFO) << "EM sub_iter=" << iter << " size=" << model_tgt.GetPieceSize()
                    << " obj=" << objective_tgt << " num_tokens=" << num_tokens_tgt
                    << " num_tokens/piece="
                    << 1.0 * num_tokens_tgt / model_tgt.GetPieceSize();
         }
         //break;

         if(model_src.GetPieceSize()<=trainer_src->desired_vocab_size_ \
                 && model_tgt.GetPieceSize()<=trainer_tgt->desired_vocab_size_)
             break;

         LOG(INFO)<<"prune called";

         PruneSentencePiecesJoint(trainer_src, trainer_tgt, model_src,model_tgt);
         PruneSentencePiecesJoint(trainer_src, trainer_tgt);
         auto ret = PruneSentencePiecesJoint();
         

         auto new_sentencepieces_src= trainer_src->PruneSentencePieces(model_src);
         auto new_sentencepieces_tgt= trainer_tgt->PruneSentencePieces(model_tgt);

         model_src.SetSentencePieces(std::move(new_sentencepieces_src));
         model_tgt.SetSentencePieces(std::move(new_sentencepieces_tgt));
     }
    LOG(INFO)<<"while break";

    trainer_src->final_pieces_ = trainer_src->FinalizeSentencePieces(model_src);
    trainer_tgt->final_pieces_ = trainer_tgt->FinalizeSentencePieces(model_tgt);

    trainer_src->Save();
    trainer_tgt->Save();


    //trainer_src->Train();
    //trainer_tgt->Train();
    




    return util::OkStatus();
}


util::Status SentencePieceAlignTrainer::PruneSentencePiecesJoint(
            const std::unique_ptr<unigram::Trainer> &trainer_src,
            const std::unique_ptr<unigram::Trainer> &trainer_tgt,
            const unigram::TrainerModel &model_src,
            const unigram::TrainerModel &model_tgt){

    LOG(INFO)<<"Align Trainer full called";
    auto new_sentencepieces_src = trainer_src->PruneSentencePieces(model_src);
     LOG(INFO)<<"sentencepieces_"<<typeid(new_sentencepieces_src).name();
    model_src.SetSentencePieces(std::move(new_sentencepieces_src));

    //auto new_sentencepieces_tgt = trainer_tgt->PruneSentencePieces(model_tgt);
    //model_tgt.SetSentencePieces(std::move(new_sentencepieces_tgt));
    return util::OkStatus();
    }

util::Status SentencePieceAlignTrainer::PruneSentencePiecesJoint(){
    return util::OkStatus();
}
util::Status SentencePieceAlignTrainer::PruneSentencePiecesJoint(
            const std::unique_ptr<unigram::Trainer> &trainer_src,
            const std::unique_ptr<unigram::Trainer> &trainer_tgt){
    return util::OkStatus();
}
}// namespace sentencepiece
 
