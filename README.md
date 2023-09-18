# gpt2-dialogue-generation-pytorch

[2023 한국컴퓨터종합학술대회 논문집](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11488165)

### I brought a multi-tone chatbot using [devjwsong/gpt2-dialogue-generation-pytorch](https://github.com/devjwsong/gpt2-dialogue-generation-pytorch) for research on the response generation model using conversation summary. 

The shell files(train, inference) below behaves differently from the [open source above](https://github.com/devjwsong/gpt2-dialogue-generation-pytorch). Take a summary as an input, not a conversation, like this [link](https://cdmon.tistory.com/34). The same goes for inference.

### Use run.ipynb rather than Powershell, bash...

---

### References of code

<a id="1">[1]</a> Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.
([http://www.persagen.com/files/misc/radford2019language.pdf](http://www.persagen.com/files/misc/radford2019language.pdf))

<a id="2">[2]</a> How to build a State-of-the-Art Conversational AI with Transfer Learning . (2019, May 9). 
([https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313))

<a id="3">[3]</a> Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). Dailydialog: A manually labelled multi-turn dialogue dataset. *arXiv preprint arXiv:1710.03957*. 
([https://arxiv.org/abs/1710.03957](https://arxiv.org/abs/1710.03957))

<a id="4">[4]</a> Rashkin, H., Smith, E. M., Li, M., & Boureau, Y. L. (2018). Towards empathetic open-domain conversation models: A new benchmark and dataset. *arXiv preprint arXiv:1811.00207*. ([https://arxiv.org/abs/1811.00207](https://arxiv.org/abs/1811.00207))

<a id="5">[5]</a> Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., & Weston, J. (2018). Personalizing dialogue agents: I have a dog, do you have pets too?. *arXiv preprint arXiv:1801.07243*. ([https://arxiv.org/abs/1801.07243](https://arxiv.org/abs/1801.07243))

<a id="6">[6]</a> Smith, E. M., Williamson, M., Shuster, K., Weston, J., & Boureau, Y. L. (2020). Can You Put it All Together: Evaluating Conversational Agents' Ability to Blend Skills. *arXiv preprint arXiv:2004.08449*. ([https://arxiv.org/abs/2004.08449](https://arxiv.org/abs/2004.08449))

### References of paper

<a id="1">[1]</a> Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., &  Sutskever,  I..    Language  Models  are  Unsupervised Multitask Learners. OpenAI Blog, 2018. ([https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf))

<a id="2">[2]</a> Mike  Lewis,  Yinhan  Liu,  Naman  Goyal,  Marjan Ghazvininejad,  Abdelrahman  Mohamed,  Omer  Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020. 
([https://arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461))

<a id="3">[3]</a> Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S.. DailyDialog:  A  Manually  Labelled  Multi-turn  Dialogue Dataset.  Proceedings  of  the  Eighth  International  Joint Conference on Natural Language Processing (Volume 1: Long Papers), 1(1), 986-995, 2017.  
([https://arxiv.org/abs/1710.03957](https://arxiv.org/abs/1710.03957))

<a id="4">[4]</a>  Emily  Dinan,  Varvara  Logacheva,  Valentin  Malykh, Alexander H. Miller, Kurt Shuster, Jack Urbanek, Douwe Kiela, Arthur Szlam, Iulian Serban, Ryan Lowe, Shrimai Prabhumoye,  Alan  W  Black,  Alexander  I.  Rudnicky, Jason  Williams,  Joelle  Pineau,  Mikhail  Burtsev,  and Samira Ebrahimi Kahou. Beyond Goldfish Memory: Long-Term  Open-Domain  Conversation.  Proceedings of  the  57th  Annual  Meeting  of  the  Association  for Computational Linguistics (ACL), 2019. 
([https://arxiv.org/abs/2107.07567](https://arxiv.org/abs/2107.07567))
