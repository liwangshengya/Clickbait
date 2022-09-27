主要修改了loss,按照T_CE的方法加入了BPR_loss的截断，目前也只在tiktok上实现了
##### 修改地方
1. 在args参数中加入了loss_type来选择loss的类型，将其传入到loss函数中
2. loss中加入了两个新参数,interation,loss_type,interation是仿照T_CE中的interation来加的，主要记录了总的batch轮数,具体统计方法就是在epoch和dataloader的两重循环中自增
3. BPR_T_CE还是仿照T_CE来写的，这里代码书写的合理性和截断的取值都有待商榷，尤其是这里还是按照T_CE来舍弃drop_rate数量的loss，重新计算剩余的loss。这里的几个参数要仔细设计一下。
``` python
# define drop rate schedule
def drop_rate_schedule(iteration, drop_rate=0.1,exponent=1,num_gradual=30000 ):
    # addr 0.05 1   30000
    #amazon 0.1 1   30000
    #yelp  0.1 1 30000
	drop_rate = np.linspace(0, drop_rate**exponent, num_gradual)
	if iteration < num_gradual:
		return drop_rate[iteration]
	else:
		return drop_rate
```
4. 这里的作者写的CE_loss和交叉熵公式有点不太一样，原始二元交叉熵loss是真实值和标签值之间的运算，这里成了副样本和正样本之间的
5. 由于这里的CE的运算还没搞懂，所以是按照那种方式来实现T_CE还要进一步研究讨论，这里暂时还没有实现
$$ Out = -Labels * \log(\sigma(Logit)) - (1 - Labels) * \log(1 - \sigma(Logit)) $$
 ```python
  def loss(self, data,interation,loss_type=1):
        user, pos_items, neg_items = data
        pos_scores, neg_scores, pre_pos_scores, pre_neg_scores = self.forward(user.to(device), pos_items.to(device), neg_items.to(device))
        if loss_type == 0:
            # BPR loss
            loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
            loss_value_pre = -torch.sum(torch.log2(torch.sigmoid(pre_pos_scores - pre_neg_scores)))
            return loss_value + self.alpha * loss_value_pre

        if loss_type == 1:
            #BPR_T_CE loss
            loss_value=-torch.log2(torch.sigmoid(pos_scores - neg_scores))
            loss_value_pre=-torch.log2(torch.sigmoid(pre_pos_scores - pre_neg_scores))
            loss=loss_value+self.alpha*loss_value_pre

            drop_rate=drop_rate_schedule(interation)

            # #去除小于dorp_rate的loss
            # loss=loss[loss>drop_rate]


            rememer_rate=1-drop_rate
            idx_loss=np.argsort(loss.cpu().detach().numpy())
            idx_loss=idx_loss[:int(len(idx_loss)*rememer_rate)]
            #取出留下的样本
            loss=loss[idx_loss]
            print('loss',loss.sum())
            return loss.sum()
 ```
