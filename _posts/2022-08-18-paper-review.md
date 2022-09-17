---
layout: post
title: "[논문리뷰] (2022) Few-shot Learning with Retrieval Augmented Language Models"
categories: paper
author:
- Eunjoo Jeon
tags: 
- deeplearning, Language Model, Few-shot learning
use_math: true
---

Izacard, Gautier, et al. "Few-shot Learning with Retrieval Augmented Language Models." arXiv preprint arXiv:2208.03299 (2022).
https://arxiv.org/pdf/2208.03299

## Preliminaries:
- LLM은 가지고있는 많은 parameter에 많은 양의 corpus로 부터 얻어낸 정보를 memorize하고, sub-task가 주어질 때 그 정보를 활용한다.  GPT3가 1750억 parameter쯤 되는데 그 많은 parameter 중에 어디에 memorize된 정보를 가지고 예측을 하는걸까? training data 중 어떤 문서에서 정답을 뽑아 낸건지에 대한 설명이 부족하다. 그래서 정답이 있는 document부터 뽑고 (Retireval) 그 문서 안에서 NLP task를 수행하는 (Q&A 등) Retrieval Augmented architecture가 나왔다.
<center> 그림1. REALM architecture </center>
<center><img src="https://user-images.githubusercontent.com/16963245/185537814-e823358f-9f96-4dbd-bfd4-ddd4701fddec.png" width="50%" height="50%"></center>
Retrieval Augmented architecture: REALM: Retrieval-Augmented Language Model Pre-Training (https://arxiv.org/abs/2002.08909)


## Abstract
Large Language model (LLM)은 여러 NLP task의 few-shot 학습 결과에서 좋은 성능을 보이고 있다. 하지만, LLM은 massive parameter가 지식을 저장하기 위해 필요하다. Retrieval augmented model은 이런 많은 수의 parameter를 필요로 하지 않는 지식 집약적 task에 적절한 방법으로 떠오르지만, few-shot setting에서 어떻게 작동하는지에 대해서는 설명이 부족했다. 

이 연구에서는 few-shot에 pre-trained retrieval augmented language model을 적용하여 Atlas라는 모델을 소개한다. MMLU, KILT, NaturalQuestions, document indexing등 여러 테스크에서 성능을 확인했다. 그 결과 Atlas는 Natural Question에서 64개 example만 가지고 42% 정확도를 내었다. 이는 540B parameter를 가진 모델보다 50배 적은 parameter만으로 3% 성능을 향상한 것이다. 

## Introduction 
LLM이 여러 언어 테스크에서 좋은 성능을 낼 수 있는 것은 1) 엄청난 크기의 parameter, 2) 엄청난 양의 training data때문이다. LLM은 복잡한 reasoning을 위한 larger computational budget이 필요하고 많은 training data에서 뽑은 정보를 memorize하기 위한 능력이 필요하다. 

하지만 few-shot learning은 in-parameter memorisation(파라미터 안에 저장) 을 한다고 말할 수 있을까? few-shot learning은 LLM의 parameter중에 어떤 정보를 활용한것일까? 

이 연구에서는 few-shot learning도 parameter에 정보를 저장하는지? 만약에 저장한다면 일반적인 learning mode과 어떻게 분리 가능한지? 를 알아본다. 본 연구에서 memory는 outsourcing가능하고, Retrieval-augmented architecutre를 통해 외부의 non-parametric knowledge source로 변경가능하다고 가정한다.

<center> 그림2. Atlas (fig1) </center>
<center><img src="https://user-images.githubusercontent.com/16963245/187019385-5b5c58ef-48aa-43ff-ab71-d7a9137d772c.png
" width="80%" height="50%"></center>

그림1에서 보는 바와 같이 기본적으로 Retrieval-augmented architecture이다. Qeury를 인풋으로 받아 1) Retrieve이 relevant한 document를 찾고 2) LM모델이 찾은 문서 안에서 Answer를 찾아 return한다. (그림1의 중앙의 아래 회색 부분이 찾아진 document를 나타낸다.)
추가적으로 MLM만 학습하고, Fact checking, Question answering은 few-shot learning을 수행하는 구조임. 

## Method
text-to-text framework: input을 text query로 (예, 'Where is the Bermuda Triangle?") output을 text answer(예, "Western part of North Atlantic Ocean")로 생성하는 구조

### 2.1 Architecture
#### 1) Retrieval
Contriever (Izacard et al., 2022; )를 문서를 retrieval하는데 사용함. Contriever는 TF-IDF, BM25같이 단어 출현에 따른 sparse metrics를 기본으로 한 검색방법이 아닌 encoder를 통해 query와 document를 embbeding하여 continous vector끼리의 거리를 계산하여 쿼리와 가까운 문서를 선택하는 dense retirever 모델이다. 

Contriever는 transformer encoder로 query와 document를 각각 encoding하고 average pooling으로 vector를 줄인다. query와 document의 최종 embedding vector끼리 dot product로 similarity를 계산하여 relevant 문서를 선택한다. Contriever의 Encoder는 unsupervised data에 대해 MoCo contrastive loss로 학습하였다. 

Contrastive loss는 아주 간단히 말해 positive pair와 negative pair를 input으로 받아 (자신의 데이터를 augmented함으로 lable은 따로 필요없음) positive는 더 가깝게 negative는 멀게 학습시키는 것이다. (자세한 내용은 https://89douner.tistory.com/334 참조)


MoCo constrastive loss는 negative sampling의 computational cost를 줄이기 위해 negative regrepsentation을 queue에 저장하고, key encoder를 momentum update하도록 변형한 것이다.  
<center> 그림3. MoCo constrastive loss </center>
<center><img src="https://user-images.githubusercontent.com/16963245/187019510-aad67e1c-73ac-469b-8e75-a36e2ac7c647.png" width="80%" height="50%"></center>

#### 2) Language model 
T5 sequence-to-sequence architecture (Raffel et al., 2019) 를 사용함. 
Fusion-in-Decoder에 기반함. Fusion-in-Decoder의 답변 생성 방식은 쿼리와 Retreival 된 문서를 입력으로 만들고 각각 인코더에 태워 vector로 만듦. 그리고 이 N개의 인코딩 백터들을 합쳐 (concatenate) 디코더에 넣고 답변을 생성함. 
본 논문에서 다른점은 "Contriever"로 검색된 문서에 query를 직접 붙이지 않고 encoder안에서 embedding하면서 붙임. 

<center> 그림4. FUSION-IN-DECODER (2)</center>
<center><img src="https://user-images.githubusercontent.com/16963245/187019130-dbbb69b9-2a8c-4094-8a1f-e18eeb96cc0e.png" width="80%" height="50%"></center>

### 2.2 Training loss 
Retriever이랑 LM을 둘다 학습시키기위해 4개의 loss 활용.

LM모델이 활용한 문서로 생성한 answer가 잘 맞으면 Retrieval한테 supervisory signal을 주어서 해당 문서를 상위 rank하게 학습시키게 함. Document relevant에 대해 labeling할 필요 없이, query랑 output pair만 있으면 됨.self-supervised나 pre-training 없어도 어떤 task든 적용 가능함. 
#### 1) Attention Distillation (ADist)
입력 문서와 output (answer)의 cross-attention score를 output이 생성될 때 각 문서의 중요도를 proximate하는데 사용. (cross-attention: encoder의 out인 K,V, decoder의 query 다른 출처의 Q,K,V cross)

이렇게 계산된 각 문서의 중요도 분포와 (LM에서 계산한)와 Retriever이 뽑은 top-K개의 문서의 분포($p_{RETR}(\textbf{d}\vert\textbf{q})$) 의 차이 (KL-divergence)를 최소화 하도록 loss 구성. 

단, 이 loss는 Retriver를 학습시키기 위한 것으로만 활용되며 LM모델에 활용되지 않음. (LM모델이 정답 내는데 활용한 document에 최대한 맞게 Retriver를 학습). $p_{ATTN}$ 은 STOPGRADIENT operator 써서 구현함. 
<center><img src="https://user-images.githubusercontent.com/16963245/187021067-23c2cbde-998e-4b42-88ea-458fc5c18649.png" width="50%" height="50%"></center>

 - $p_{RETR}$ : encoding된 doc과 query vector의 doct-product (s), theta 는 temperature hyper-parameter
 <center><img src="https://user-images.githubusercontent.com/16963245/187021157-fe6efd24-3e70-446f-9fb5-1f297d126788.png" width="50%" height="50%"></center>

 - $p_{ATTN}$ : LM의 cross-attention에서 attention score와 value가지고 계산. $a_{n}\lVert v_{n} \rVert 2$값을 한 document에 대한 모든 attention head, layer, tokens의 평균을 구함. SOFTMAX operator를 통해 구한 값의 distribution $p_{ATTN}(d_{k})$를 구함. 

#### 2) End-to-end training of Multi-Document Reader and Retriever (EMDR^2)
기대값 최대화 (expectation-maximization algorithm)에서 영감을 받아 검색된 문서를 latent variable로 취급한다.  
<center><img src="https://user-images.githubusercontent.com/16963245/190842129-b988f634-03d8-4c4d-b4ff-746ba616741c.png" width="50%" height="50%"></center>

$p_{RETR}$는 ADist loss에서와 같이 검색된 top-K 문서에 대한 분포이고, $p_{LM}$ 과 비교하여 Retriever만 학습한다. 
$p_{LM}$은 query, top K 문서가 주어졌을때, 해당 출력(a)의 확률로써, 결국 EMDR^2를 최대화 하는 document K를 잘 뽑도록 $p_{RETR}$을 학습시킨다. 

#### 3) Perplexity Distillation (PDist)
LM모델의 혼란도(perplexity)를 줄이도록 top K문서를 잘 뽑을 수 있게 Retriever를 학습. 
<center><img src="https://user-images.githubusercontent.com/16963245/190842445-4e0d7cd8-43f8-4c8d-b09c-08fe8c0bb2fa.png" width="50%" height="50%"></center>
Retriever의 문서 분포와 LM모델에 의한 문서의 사후 확률의 분포 차이 (KL divergence)를 최소 화 시킴. 

#### 4) Leave-one-out Perplexity Distillation (LOOP)
마지막으로 검색된 상위 k개 문서 중 하나를 제거할 때 언어 모델의 예측이 얼마나 나빠지는지를 기반으로 LOSS를 제안. 
<center><img src="https://user-images.githubusercontent.com/16963245/190842646-deaa3422-2258-46bd-902b-e3d229720d9a.png" width="50%" height="50%"></center>

각각의 k-1 문서의 log probability 계산함. 이를 위해 k-1 문서의 각 하위 집합에 대한 출력의 로그 확률을 계산하고 음수 값을 각 문서의 관련성 점수로 사용함. softmax 연산자를 사용하여 문서에 대한 확률 분포를 획득.
그런 다음 이 분포와 리트리버로 얻은 분포 사이의 KL-divergence 최소화합니다.

### 2.3 Pretext tasks
Retreiver과 LM을 동시에 학습하기 위한 un-supervised learning task 
- Prefix language modeling: chunk of N words를 2개로 나누고 (N/2로 길이 동일하게), 첫번째 sub-sequence를 query로 나머지를 정답으로함. 1번째 문단으로 문서를 검색하고, 언어모델이 2번째 문단을 생성. 
- Masked language modeling: chunk of N words에서 k개의 span (평균 3token)을 추출하고, 그 안의 15%를 masking하고 다른 특정 token으로 채워 넣음. masked query로 문서를 검색하고, 언어 모델은 masked span을 생성. 
- Title to section generation: Wikipedia article과 section title을 입력으로 section 내용을 생성하는 task. 
- 
### 2.4 Efficient retriever fine-tuning
Retrieval은 문서에 대한 index를 활용하여 검색이 빨라질 수 있다. 이 연구에서 retreiver과 LM을 같이 학습시키면 문서에 대한 index를 매번 업데이트 시켜주어야한다. 많은 문서의 index에 대한 업데이틑 computational expensive. 이를 해결하기 위한 효율적인 방법을 제안
- Full index update: (계산식은 생략) 3,700만 개의 문서(위키피디아 인덱스의 크기)가 포함된 인덱스를 사용한다면, batch size 64로 20개의 검색된 문서를 매번 1000 step마다 index를 refresh 시킨다면, 30%의 오베데드 발생 
- Re-ranking: K*10개 정도 큰 범위의 L개 문서 (전체 문서보단 적은)에 대해서 re-embedding하고, 그 안에서 reranking을 통해 TopK문서 추출. 10%의 오버헤드 발생하지만 계속 학습되는 가운데 L개 문서 중에서 TopK가 항상 있을 거란 보당이 없고, L개 문서 중에 Top K를 다시 뽑는 re-ranking도 하나의 추가 task이다. 
- Query-side fine-tuning: 마지막 전략은, 인코딩 된 쿼리와 문서를 분리 시킴. 문서 인코더는 fix하고 쿼리 인코더만 학습. 문서 임베딩은 고정임으로 index업데이트가 필요없으므로 오버헤드는 미 발생. (Retreiver학습 다 끝난 다음에 다시 indexing하는 듯?) 

## Experiment 
### 4.1 Benchmarks
- Knowledge-Intensive Launguage Tasks(KILT): 11 datasets 5 tasks; QA (NaturalQuestion, TriviaQA, HotpotQA), slot filling (Zero Shot RE, T-REx), entity linking (AIDA, CoNLL-YAGO), dialog (Wizard of Wikipedia), fact checking (FEVER)
- Massively-Mutitask Language Understanding (MMLU): 57 multi-choice question anwering datasets. 각 도메인에 대해 zero-shot, multi-task few-shot, transfer learning (다른 multi-choice QA task로 학습한 걸 test)
- Additional benchmarks: open-domain NaturalQuestions, TriviaQA. TempLAMA (time-sensitive cloze question; 2017-2020)
### 4.2 Technical details 
- Pre-training: Retriever는 BERT-based contirever initialize, LM은 unlabed text로만 학습된 T5 pretrained weight 활용 (있는거). Retrieve document는 20개. 
- Fine-tuning: downstream task수행을 위해 fine-tuning수행. ablation study를 위해 iteration 고정. 50 iter (64-shot), 200 iter(1024-shot). 
- Unlabed datasets: Wikipedia dump 사용 (22.12.20 ver). section 별로 나누고, 긴 section은 200 words이상 포함하게 동일 size의 passage로 자름. 37M passage (평균 78 words) 데이터 확보. crawl dmp (20.10.00 ver) 활용. 총 350M passage. 

### 4.3 Pre-training loss and tasks

- RQ1. Retriever과 LM 동시에 학습 (jointly)시키면 few-shot 성능이 올라가나? 
- 네
- RQ2. Retriever 학습 시키기 위한 가장 좋은 loss function은? 
- 4가지 통계적 차이 없음. 그래서 Perplexity Distillation 선택 (ADist, EMDR보다 안정적이고, LOOP보다 계산량 적음)


<center><img src="https://user-images.githubusercontent.com/16963245/190844719-f681154d-045e-4d00-ba56-a08088c4713c.png" width="90%" height="50%"></center>
Table1을 보면, Closed-book (non-augmented T5)가 가장 성능이 안좋음. No joint와 그 아래 joint를 비교하면 Joint가 상대적으로 성능 좋음. 다만 MLM외에는 Fiexed retreiver과 4개의 loss로 학습 시킨 것 간의 차이가 별로 없음. few-shot task는 LM에 가장 큰 영향을 받는다. 4개의 loss들 간에는 크게 차이 없음. 

Pretraining은 MLM이 성능이 좋아서 그걸로 선택, Index (Wiki, Commoncraw)와 Training data(Wiki, CC) 조합에 대한 설명은 생략 (실험적으로 조합 선택)
### 4.4 Fine-tuning
- RQ3. 제한된 학습 데이터로 Atlas를 어떻게 효과적으로 fine-tune시킬 것인가? 
<center><img src="https://user-images.githubusercontent.com/16963245/190845044-4c93b72d-0aae-4943-a685-5089e7d2fc64.png" width="90%" height="50%"></center>

64-, 1024-shot을 위해 fine-tuning할 때 retreiver를 fiex해 두는것은 성능 저하를 일으켰다. 앞서 제안한 방법 중 re-ranking 방식은 full-text update와 비슷한 성능을 보였고, Query-side fine-tuning은 64-shot에서 성능이 좋았다. example이 적을 때는 Query-side fine tuning을 example이 많을 때에는 Standard fine-tuning을 활용함. 
### 4.5 Training and evaluating Atlas 
#### 1) MLM 
<center><img src="https://user-images.githubusercontent.com/16963245/190845598-a701f889-e229-425f-838d-c9b52a19f53c.png" width="90%" height="50%"></center>
T5만 쓴거는 4지선다 MMLU의 random 성능(25%)를 겨우 상위한다. Atlas는 770M parameter만 사용해도 40%정도 성능이 나옴.
데이터가 많아 질 수록 (5-shot -> 5shot multi-task -> full) 모두 성능은 좋아지나, Atlas만 multi-task 50shot에서 성능이 떨어지는데 이는 paremeter가 적어서 여러 task끼리 synerge를 내게 학습하기 어려웠을 것으로 생각된다. 
데이터가 많아 질 수록 성능은 모두 향상되나, 그 gap은 유지 되어 Atlas의 성능 LM만 쓸때보다 좋음을 알 수 있다. 

<center><img src="https://user-images.githubusercontent.com/16963245/190845716-9ba0f5d3-608b-44e4-864e-225ce5d62dda.png" width="90%" height="50%"></center>
zero-shot은 random보다 좋음. 
다른 LM모델 (GPT-3, Chincilla), RAG (Gopher)과 비교했을때 GPT3보단 좋지만 Chicilla보단 떨어짐. Full learning으로 가면 Atlas가 모델 parameter가 적음에도 성능이 좋음. 

#### 2) QA 
Atlas는 특히 QA 테스크에서 높은 성능을 보임. Retreiver-augmented architecture의 장점을 보여줌. 

GPT-3, PaLM등 거대 언어 모델 대비 높은 성능을 보여줬으며, RA 모델인 Gopher (50 pasage retreive한 후, 4개의 answer 생성, re-ranking 통해 최종 answer선택) 보다 25배 적은 파라미터인데도 성능이 더 좋음. 
<center><img src="https://user-images.githubusercontent.com/16963245/190845826-558a5317-f9e4-4532-bb7d-aa13af4c166c.png" width="90%" height="50%"></center>

#### 4) FEVER 
15-shot에서 Gopher보다 5.1높은 56.2% 성능 보임. full learning에서 ProoFVer (sentence-leve annotation으로 retriever를 학습) 보다 1.5% 낮지만, PrroFVer가 학습한데로 FEVER가 제시한 Wiki corpus로 학습하면 (Atlas는 CCNet의 wiki corpus로 학습) 80.1%로 SOTA임. 
<center><img src="https://user-images.githubusercontent.com/16963245/190846078-08f38b6c-0744-423b-bc24-e94b7d7bdee8.png" width="90%" height="50%"></center>

#### 5) KILT 
Atlas가 64-shot setting에서 ADIA는 random 상회, FEVER 는 SOTA대비 2-2.5포인트 밑, Zeroshot RE는 SOTA들 보다 높음 

Full train setting에서는 T-REx, zsRE, TQA 3개 빼고 SOTA고 3개도 다른 SOTA대비 3% 내 정확도. 
<center><img src="https://user-images.githubusercontent.com/16963245/190846231-6aae58e0-60a8-44e3-a150-0dba6c34b4c5.png" width="90%" height="50%"></center>

## 느낀점 
아. 길어서 읽느라 힘들었다. 

Retrieval-augmented architecture에 대해 이해할 수 있었고, 제시한 loss 4개가 재밌었다. 결과는 언제나 논문결과가 그렇듯 이리 저리 condition을 바꿔서 SOTA대비 더 좋다라고 나왔지만 REALM논문이 few-shot setting에 대해 연구를 안한거말고 크게 차이가 있는지는 REAML을 읽어봐야겠다. 

