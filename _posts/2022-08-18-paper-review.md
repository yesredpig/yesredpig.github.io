---
layout: post
title: "[논문리뷰] (2022) Few-shot Learning with Retrieval Augmented Language Models"
categories: paper
author:
- Eunjoo Jeon
---

Izacard, Gautier, et al. "Few-shot Learning with Retrieval Augmented Language Models." arXiv preprint arXiv:2208.03299 (2022).
https://arxiv.org/pdf/2208.03299

## Preliminaries:
- #### Retrieval Augmented architecture: REALM: Retrieval-Augmented Language Model Pre-Training (https://arxiv.org/abs/2002.08909)
- <img height="450" src="C:\Users\LG\PycharmProjects\fig1.png" width="300"/>
- LLM이 가지고있는 parameter에 memorize한다는건 GPT3 처음 나왔을때 많은 discussion이 있었던 걸로 안다. GPT3 (이젠 오래된 것 같은데)가 1750억 parameter이고 데이터는 한 5000억 쯤 되는걸로 안다. 문제는 그 많은 parameter 중에 어디에 memorize된 정보를 가지고 예측을 하냐는거다. 어떤 문서에서 정답을 뽑아 낸건지에 대한 설명이 부족하다. 그래서 정답이 있는 document부터 뽑고 (Retireval) 그 안에서 NLP task를 수행하는 (Q&A 등) Retrieval Augmented architecture가 나온 것이다.

- #### few-shot learning
- 이 논문에서 말하는 few-shot은 아마도 GPT3에서 예제로 나온 few-shot등을 말하는 것 같다. GPT3논문을 봐도 왜 few-shot이 잘나온지는 잘 모르겠었는데, 그걸 설명하려는 것 같다.


## Abstract
Large Language model (LLM)은 여러 NLP task의 few-shot 학습 결과에서 좋은 성능을 보이고 있다. 하지만, qeustion answering, fact checking등의 task를 LLM이 잘 수행하는데 있어 massive parameter가 지식을 저장하기 위해 필요하다. Retrieval augmented model은 이런 많은 수의 parameter를 필요로 하지 않는 지식 집약적 task에 적절한 방법으로 떠오르지만, few-shot setting에서 어떻게 작동하는지에 대해서는 설명이 부족했다. 

이 연구에서는 few-shot에 pre-trained retrieval augmented language model을 적용하여 Atlas라는 모델을 소개한다. MMLU, KILT, NaturalQuestions, document indexing등 여러 테스크에서 성능을 확인했다. 그 결과 Atlas는 Natural Question에서 64개 example만 가지고 42% 정확도를 내었다. 이는 50배 적은 parameter만으로 540B parameter를 가진 모델보다 3% 성능이 높은 것이다. 

## Introduction 
LLM이 여러 언어 테스크에서 좋은 성능을 낼 수 있는 것은 1) 엄청난 크기의 parameter, 2) 엄청난 양의 training data때문이다. LLM은 복잡한 reasoning을 위한 larger computational budget이 필요하고 많은 training data에서 뽑은 정보를 memorize하기 위한 능력이 필요하다. 

하지만 few-shot learning은 in-parameter memorisation을 한다고 말할 수 있을까? few-shot learning은 LLM의 parameter중에 어떤 정보를 활용한것일까? 

이 연구에서는 few-shot learning도 information을 자신들의 parameter에 저장하는지? 만약에 저장한다면 일반적인 것과 어떻게 분리 가능한지? 를 알아본다. 본 연구에서는 memory는 outsourcing가능하고, Retrieval-augmented architecutre를 통해 외부의 non-parametric knowledge source로 변경가능하다고 가정한다.

## Method

### Some great subheading (h3)
