import time
from copy import deepcopy

import torch
import torch.optim as optim
import numpy as np
from Utils.dcd_evaluation import evaluation, print_final_result
from Utils.dcd_loss import relaxed_ranking_loss
from pdb import set_trace as bp
import numpy as np



def DCD(opt, model, gpu, optimizer, train_loader, test_dataset, I_dataset, model_save_path, user_count, item_count, user_topk, item_topk):
  
  max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch

  save = False
  if model_save_path != None:  save= True

  train_loss_arr = []
  template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
  eval_dict = {5: deepcopy(template), 10:deepcopy(template), 20:deepcopy(template), 'early_stop':0,  'early_stop_max':early_stop, 'final_epoch':0}


  total_items = torch.LongTensor([i for i in range(item_count)]).to(gpu)
  total_users = torch.LongTensor([i for i in range(user_count)]).to(gpu)
  for_rank = torch.LongTensor([r for r in range(500)]).to(gpu)
  for_rank_item = torch.LongTensor([r for r in range(500)]).to(gpu)
  tic_cal_matrix = time.time()
  user_topk_dict = user_topk
  item_topk_dict = item_topk

  tmp_user = torch.Tensor(user_topk_dict[0]).unsqueeze(0)  # teacher user x top-k item, each element is index of item
  for user_idx in range(1,user_count):
    tmp_user = torch.cat([tmp_user,torch.Tensor(user_topk_dict[user_idx]).unsqueeze(0)],dim=0)


  tmp_item = torch.Tensor(item_topk[0]).unsqueeze(0)  # teacher item x top-k user
  for item_idx in range(1,item_count):
    tmp_item = torch.cat([tmp_item,torch.Tensor(item_topk[item_idx]).unsqueeze(0)],dim=0)  



  with torch.no_grad():


    U = torch.zeros((user_count, item_count)) ## teacher ,each element is rank
    I = torch.zeros((item_count, user_count)) ## teacher , each element is rank
    for i in user_topk_dict:
      cnt = 0
      for u in user_topk_dict[i]:
        U[i][u] = cnt
        cnt+=1
    del user_topk_dict
    for i in item_topk_dict:
      cnt = 0
      for u in item_topk_dict[i]:
        I[i][u] = cnt
        cnt+=1
    del item_topk_dict

  ###item -side
  for epoch in range(max_epoch):
    tic1 = time.time()

    if epoch % 3== 0:
      with torch.no_grad():
        rank_diff = torch.zeros( (user_count,item_count))
        rank_diff_item = torch.zeros( (item_count,user_count))
        rank_diff_500 = torch.zeros( (user_count,500))
      
        rank_diff_500_inv = torch.zeros( (user_count,500))
        for k in range(0,user_count,1000):
          if k ==0:
            score =  model.forward_multi_items(total_users[k:(k+1000)], tmp_user[total_users[k:(k+1000)]].type(torch.LongTensor).to(gpu)) 
          elif k+1000 < user_count:
            score_tmp = model.forward_multi_items(total_users[k:(k+1000)], tmp_user[total_users[k:(k+1000)]].type(torch.LongTensor).to(gpu))
            score = torch.cat((score,score_tmp) ,0)
          else :
            score_tmp = model.forward_multi_items(total_users[k:], tmp_user[total_users[k:]].type(torch.LongTensor).to(gpu))
            score = torch.cat((score,score_tmp) ,0)
            del score_tmp
        s_rank = torch.argsort(-score) # index 1 0 2
        s_rank = torch.gather(tmp_user[total_users].to(gpu), 1,s_rank.to(gpu)).type(torch.LongTensor) 
        # example: tmp_user : [3,2,1] teacher rec list, score : [0.3, 0.1 , 0.6] score of tmp_user


        rank_diff_500_item = torch.zeros( (item_count,500))
        rank_diff_500_item_inv = torch.zeros( (item_count,500))
        for k in range(0,item_count,1000):
          if k ==0:
            score_item =  model.forward_multi_users(tmp_item[total_items[k:(k+1000)]].type(torch.LongTensor).to(gpu), total_items[k:(k+1000)])
          elif k+1000 <  item_count:
            score_item_tmp =  model.forward_multi_users(tmp_item[total_items[k:(k+1000)]].type(torch.LongTensor).to(gpu), total_items[k:(k+1000)])
            score_item = torch.cat((score_item,score_item_tmp) ,0)
          else :
            score_item_tmp =  model.forward_multi_users(tmp_item[total_items[k:]].type(torch.LongTensor).to(gpu), total_items[k:])
            score_item = torch.cat((score_item,score_item_tmp) ,0)
            del score_item_tmp
        s_rank_item = torch.argsort(-score_item)
        s_rank_item = torch.gather(tmp_item[total_items].to(gpu), 1,s_rank_item.to(gpu)).type(torch.LongTensor)
        # item x 200(top-k) , each element is index, student rec list using teach top-k user
        tmp_user = tmp_user.type(torch.LongTensor)
        tmp_item = tmp_item.type(torch.LongTensor)

        for s in range(user_count):
          S = torch.zeros(item_count).type(torch.LongTensor).to(gpu)
          S[s_rank[s]] = for_rank  # each element is rank, 1 x item  ex, [1, 3, 2]  item 0 is rank 1, item 1 is rank 3
          diff_tmp = S - U[s].to(gpu)  # rank diff student - teacher
          diff_tmp_inv = U[s].to(gpu) - S  # rank diff teacher - student
          rank_diff[s] = torch.tanh(torch.maximum(diff_tmp*opt.ep,torch.zeros_like(diff_tmp)))  # undereestimation error matrix
          rank_diff_500[s] = rank_diff[s][tmp_user[s]]  #error sorting by teacher rec list, each element is error, but rec list
          rank_diff_500_inv[s] = torch.tanh(torch.maximum(diff_tmp_inv*opt.ep,torch.zeros_like(diff_tmp)))[tmp_user[s]] # overestimation ereror, sorted by teacher rec list
        del S
        del diff_tmp
        del diff_tmp_inv


        for s in range(item_count):
          S_item = torch.zeros(user_count).type(torch.LongTensor).to(gpu)
          S_item[s_rank_item[s]] = for_rank_item
          diff_tmp_item = S_item - I[s].to(gpu)
          diff_tmp_item_inv = I[s].to(gpu) - S_item
          rank_diff_item[s] = torch.tanh(torch.maximum(diff_tmp_item*opt.ep,torch.zeros_like(diff_tmp_item)))# undereestimation error matrix
          rank_diff_500_item[s] = rank_diff_item[s][tmp_item[s]]#error sorting by teacher rec list, each element is error, but rec list
          rank_diff_500_item_inv[s] = torch.tanh(torch.maximum(diff_tmp_item_inv*opt.ep,torch.zeros_like(diff_tmp_item)))[tmp_item[s]]# overestimation ereror, sorted by teacher rec list
        del S_item
        del diff_tmp_item
        del diff_tmp_item_inv
        
    train_loader.dataset.negative_sampling()
    train_loader.dataset.U_sampling()
    I_dataset.I_sampling()
    epoch_loss = []
    for batch_user, batch_pos_item, batch_neg_item in train_loader:
      
      
      # Convert numpy arrays to torch tensors
      batch_user = batch_user.to(gpu)
      batch_pos_item = batch_pos_item.to(gpu)
      batch_neg_item = batch_neg_item.to(gpu)
      
      # Forward Pass
      model.train()

      # Base Loss
      output = model(batch_user, batch_pos_item, batch_neg_item)
      base_loss = model.get_loss(output)
      batch_user = batch_user.unique()


      with torch.no_grad():

          for_p_500 = rank_diff_500[batch_user]
          a_1 = torch.multinomial(for_p_500[:len(for_p_500)//2],opt.p_n) 
          a_2 = torch.multinomial(for_p_500[len(for_p_500)//2:],opt.p_n)

          a = torch.cat([a_1, a_2], 0)
          a = a.sort(dim=1)[0] # sorting by item index
          
          batch_items = torch.gather(tmp_user[batch_user], 1, a) # sampled underestimated item index, sorted by student rec list [3,2], a is [1,0] but sampled items are [3,2]
          # s_rank: user x 200(top-k) , each element is index, student rec list using teach top-k items. [3,2,4 same in rec list above]

          batch_item =  batch_pos_item.unique().to(gpu)


          for_p_500_inv = rank_diff_500_inv[batch_user]
          a_1 = torch.multinomial(for_p_500_inv[:len(for_p_500_inv)//2],opt.p_n)
          a_2 = torch.multinomial(for_p_500_inv[len(for_p_500_inv)//2:],opt.p_n)

          a = torch.cat([a_1, a_2], 0)
          
          batch_items_inv = torch.gather(tmp_user[batch_user], 1, a) # sampled overestimated items


      with torch.no_grad():

          for_p_500_item = rank_diff_500_item[batch_item]
          a_1 = torch.multinomial(for_p_500_item[:len(for_p_500_item)//2],opt.p_n)
          a_2 = torch.multinomial(for_p_500_item[len(for_p_500_item)//2:],opt.p_n)

          a = torch.cat([a_1, a_2], 0)
          a = a.sort(dim=1)[0]
          
          batch_user_itemside = torch.gather(tmp_item[batch_item], 1, a).type(torch.LongTensor).to(gpu)

          for_p_500_item_inv = rank_diff_500_item_inv[batch_item]
          a_1 = torch.multinomial(for_p_500_item_inv[:len(for_p_500_item_inv)//2],opt.p_n)
          a_2 = torch.multinomial(for_p_500_item_inv[len(for_p_500_item_inv)//2:],opt.p_n)

          a = torch.cat([a_1, a_2], 0)
          batch_user_itemside_inv = torch.gather(tmp_item[batch_item], 1, a).type(torch.LongTensor).to(gpu)
      
      for_pos_items = batch_items.type(torch.LongTensor).to(gpu) # underestimated items
      for_neg_items = batch_items_inv.type(torch.LongTensor).to(gpu) # overestimated items

      for_pos_prediction = model.forward_multi_items(batch_user, for_pos_items)
      for_neg_prediction = model.forward_multi_items(batch_user, for_neg_items)

      UCD_loss = relaxed_ranking_loss(for_pos_prediction, for_neg_prediction)


        
  
      interesting_user_prediction = model.forward_multi_users(batch_user_itemside, batch_item)
      uninteresting_user_prediction = model.forward_multi_users(batch_user_itemside_inv, batch_item)

      ICD_loss = relaxed_ranking_loss(interesting_user_prediction, uninteresting_user_prediction)

      # batch loss
      batch_loss = base_loss + opt.ICD_lmbda * ICD_loss + opt.UCD_lmbda * UCD_loss
      epoch_loss.append(batch_loss)
      
      # Backward and optimize
      optimizer.zero_grad()
      batch_loss.backward()
      optimizer.step()
      
    epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
    train_loss_arr.append(epoch_loss)

    toc1 = time.time()
    
    # evaluation
    if epoch < es_epoch:
      verbose = 25
    else:
      verbose = 1

    if epoch % verbose == 0:
      is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
      #LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
        
      if is_improved:
        if save:
          dict_tmp = deepcopy(model.state_dict())

    if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
      break

  print("BEST EPOCH::", eval_dict['final_epoch'])
  if save:
    torch.save(dict_tmp, model_save_path)

  print_final_result(eval_dict)