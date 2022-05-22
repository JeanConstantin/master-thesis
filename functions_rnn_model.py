def tokenize(list_arg, spacy_model):
  list_token = []
  for elem in tqdm(list_arg, total=len(list_arg)):

    token_arg1 = []
    doc1 = spacy_model(elem[0])
    for token in doc1:
      token_arg1.append(token.text)
    
    token_arg2 = []
    doc2 = nlp(elem[1])
    for token in doc2:
      token_arg2.append(token.text)
    
    elem_list_token = [token_arg1, token_arg2]
    list_token.append(elem_list_token)

  return list_token

###########################################################################

def build_vocab_ft(document_list, lm):
  word_list = []
  word_dic = {}
  for doc in document_list:
    list_token1 = [item for sublist in doc for item in sublist]
    list_token2 = [item for sublist in list_token1 for item in sublist]

    word_list.append(list_token2)
    
  word_list = [item for sublist in word_list for item in sublist]
  word_list = list(set(word_list))
  print('Model size: {}'.format(str(len(word_list))))

  lm_dim = lm.get_word_vector(word_list[0]).shape[0]
  vectors = np.zeros(shape=(len(word_list)+1, lm_dim))

  for i, word in enumerate(word_list):
    word_dic[word] = i
    vectors[i, :] = lm.get_word_vector(word)
  
  #Adding pad vector = 0
  word_dic['####'] = len(word_list)
  
  return word_dic, vectors

###########################################################################

def encode_ft(list_token, word_dic, length):
  encoded_token_all = []

  for elem in list_token:
    encoded_arg1 = []
    for token in elem[0]:
      encoded_arg1.append(word_dic[token])
    if len(encoded_arg1) > length:
      encoded_arg1 = encoded_arg1[0:length]
    if len(encoded_arg1) < length:
      for i in range(length - len(encoded_arg1)):
        encoded_arg1.append(word_dic['####'])
    
    encoded_arg2 = []
    for token in elem[1]:
      encoded_arg2.append(word_dic[token])
    if len(encoded_arg2) > length:
      encoded_arg2 = encoded_arg2[0:length]
    if len(encoded_arg2) < length:
      for i in range(length - len(encoded_arg2)):
        encoded_arg2.append(word_dic['####'])

    encoded_token_all.append([encoded_arg1, encoded_arg2])
  
  return encoded_token_all

###########################################################################

def eval_model(model, x_test, y_test, batch_size):
  model.eval()

  batch_indices = list(range(len(x_test)//batch_size))
  all_predictions = np.array([])

  for b_idx in batch_indices:
    batch_start = b_idx * batch_size
    batch_end = (b_idx + 1) * batch_size
    data_batch = x_test[batch_start : batch_end]

    batch_predictions = model(data_batch).squeeze()
    predictions_class = np.argmax(batch_predictions.detach().cpu().numpy(), axis=-1)
    all_predictions = np.append(all_predictions, predictions_class)
  
  y_true = y_test.detach().cpu().numpy()[0:len(all_predictions)]
  f1 = f1_score(y_true, all_predictions, average=None)
    
  return (f1[1] + f1[2])/2

###########################################################################

def train_model(model, x_train_, y_train_, x_valid_, y_valid_, batch_size, num_epochs, lr, wd):
    print("Training...")
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    crossentloss = nn.CrossEntropyLoss(reduction='mean')

    loss_history = []
    val_loss_history = []

    f1_history = []
    val_f1_history = []
    
    batch_indices = list(range(len(x_train_)//batch_size))

    shuffle(batch_indices)

    for epoch in range(num_epochs):
        print('Currently training epoch {} of {}'.format(epoch, num_epochs))

        #############################################################
        ######Training###############################################

        model.train()
        running_train_loss = 0

        for b_idx in tqdm(batch_indices, total=len(batch_indices)):
            optimizer.zero_grad()
            
            # Create batch
            batch_start = b_idx * batch_size
            batch_end = (b_idx + 1) * batch_size
            data_batch = x_train_[batch_start : batch_end]
            y_batch = y_train_[batch_start : batch_end]
            
            # Forward pass
            predictions = model(data_batch).squeeze()
            
            # Calculate loss
            loss = crossentloss(predictions, y_batch)
            running_train_loss += float(loss)

            # Backward pass and update
            loss.backward()
            optimizer.step()

        train_loss = running_train_loss / len(batch_indices)
        train_f1 = eval_model(model, x_train_, y_train_, batch_size)

        #############################################################
        #####Evaluation##############################################

        batch_indices_val = list(range(len(x_valid_)//batch_size))
        model.eval()

        running_val_loss = 0
        running_val_confusion = np.zeros((2,2))

        crossentloss_val = nn.CrossEntropyLoss(reduction='mean')

        for b_idx_val in batch_indices_val:
          batch_start_val = b_idx_val * batch_size
          batch_end_val = (b_idx_val + 1) * batch_size
          data_batch_val = x_valid_[batch_start_val : batch_end_val]
          y_batch_val = y_valid_[batch_start_val : batch_end_val]

          predictions_val = model(data_batch_val).squeeze()

          loss_val = float(crossentloss_val(predictions_val, y_batch_val))
          running_val_loss += loss_val
        
        val_loss = running_val_loss / len(batch_indices_val)
        val_f1 = eval_model(model, x_valid_, y_valid_, batch_size)

        loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        if np.isnan(train_f1)==False:
          f1_history.append(train_f1)
        else:
          f1_history.append(0)
        if np.isnan(val_f1)==False:
          val_f1_history.append(val_f1)
        else:
          val_f1_history.append(0)

        print('Loss is {} on train set and {} on validation set \n'.format(train_loss, val_loss))
        print('F1 is {} on train set and {} on validation set \n'.format(train_f1, val_f1))

    print('Training done.')
    
    return loss_history, val_loss_history, f1_history, val_f1_history

###########################################################################

def predict(model, x_test, batch_size):
  model.eval()

  batch_indices = list(range(len(x_test)//batch_size))
  all_predictions = np.array([])

  for b_idx in batch_indices:
    batch_start = b_idx * batch_size
    batch_end = (b_idx + 1) * batch_size
    data_batch = x_test[batch_start : batch_end]

    batch_predictions = model(data_batch).squeeze()
    batch_predictions = np.argmax(batch_predictions.detach().cpu().numpy(), axis=-1)

    all_predictions = np.append(all_predictions, batch_predictions)
    
  return all_predictions