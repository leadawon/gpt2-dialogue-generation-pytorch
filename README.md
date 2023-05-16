# gpt2-dialogue-generation-pytorch

https://github.com/devjwsong/gpt2-dialogue-generation-pytorch

### I brought a multi-tone chatbot using [[https://github.com/devjwsong](https://github.com/devjwsong/gpt2-dialogue-generation-pytorch)](devjwsong/gpt2-dialogue-generation-pytorch) for research on the response generation model using conversation summary. 

The shell files(train, inference) below behaves differently from the [https://github.com/devjwsong/gpt2-dialogue-generation-pytorch](open source above). Take a summary as an input, not a conversation, like this [https://cdmon.tistory.com/34](link). The same goes for inference.

Use run.ipynb rather than Powershell, bash...
---

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Download & Preprocess all datasets.

   ```shell
   sh exec_load_data.sh
   ```

   After running it, you will have the following data directory structure if you follow the default argument setting.

   ```
   data
   └--gpt2
       └--train_utters.pickle
       └--train_ids.pickle
       └--valid_utters.pickle
       └--valid_ids.pickle
   ```

   <br/>

3. Run the following command to train the model.

   If you want to train it starting from a specific checkpoint, add the argument `ckpt_name` and make sure to notify the proper checkpoint name.

   ```shell
   sh exec_train.sh
   ```
   
   <br/>

4. Run below command to conduct an inference with the trained model.

   This time, you are required to give a specific `ckpt_name`.

   ```shell
   sh exec_infer.sh
   ```

<br/>

---

### References

<a id="1">[1]</a> Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.([http://www.persagen.com/files/misc/radford2019language.pdf](http://www.persagen.com/files/misc/radford2019language.pdf))

<a id="2">[2]</a> How to build a State-of-the-Art Conversational AI with Transfer Learning . (2019, May 9). ([https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313))

<a id="3">[3]</a> Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). Dailydialog: A manually labelled multi-turn dialogue dataset. *arXiv preprint arXiv:1710.03957*. ([https://arxiv.org/abs/1710.03957](https://arxiv.org/abs/1710.03957))

<a id="4">[4]</a> Rashkin, H., Smith, E. M., Li, M., & Boureau, Y. L. (2018). Towards empathetic open-domain conversation models: A new benchmark and dataset. *arXiv preprint arXiv:1811.00207*. ([https://arxiv.org/abs/1811.00207](https://arxiv.org/abs/1811.00207))

<a id="5">[5]</a> Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., & Weston, J. (2018). Personalizing dialogue agents: I have a dog, do you have pets too?. *arXiv preprint arXiv:1801.07243*. ([https://arxiv.org/abs/1801.07243](https://arxiv.org/abs/1801.07243))

<a id="6">[6]</a> Smith, E. M., Williamson, M., Shuster, K., Weston, J., & Boureau, Y. L. (2020). Can You Put it All Together: Evaluating Conversational Agents' Ability to Blend Skills. *arXiv preprint arXiv:2004.08449*. ([https://arxiv.org/abs/2004.08449](https://arxiv.org/abs/2004.08449))
