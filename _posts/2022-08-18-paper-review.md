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

### 2.3 Pretext tasks

### 2.4 Retriever fine-tuning

## Experiment 
### 4.1 Benchmarks

### 4.2 Technical details 

### 4.3 Pre-training loss and tasks

### 4.4 Fine-tuning

### 4.5 Training and evaluating Atlas 


### Some great subheading (h3)

## REFERENCE

- Retrieval Augmented architecture: REALM: Retrieval-Augmented Language Model Pre-Training (https://arxiv.org/abs/2002.08909)
