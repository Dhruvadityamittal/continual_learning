# Wisdm
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset = 'pamap'
# Wisdm
if(dataset == 'wisdm'):
    data_x = np.load(r"C:\Users\Dhruv\Downloads\Compressed\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\wisdm-dataset\wisdm_30hz_w10\X.npy")
    data_y= np.load(r"C:\Users\Dhruv\Downloads\Compressed\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\wisdm-dataset\wisdm_30hz_w10\Y.npy")
    data_p = np.load(r"C:\Users\Dhruv\Downloads\Compressed\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\wisdm-dataset\wisdm_30hz_w10\pid.npy")
    data_time = np.load(r"C:\Users\Dhruv\Downloads\Compressed\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\wisdm-dataset\wisdm_30hz_w10\time.npy")
    scenario1_step_subjects = [personid for personid in np.unique(data_p) if str(personid) not in ['1616', '1618', '1637', '1639', '1642']]

    le = LabelEncoder()
    le.fit(np.unique(data_y))
    data_y = le.transform(data_y)
    
    initial_step_classes = np.unique(data_y)[0:14]
    incremental_step_classes = np.unique(data_y)[14:]
if(dataset == 'realworld'):
    data_x = np.load(r"C:\Users\Dhruv\Downloads\Compressed\RealWorld\realworld_30hz_w10\X.npy")
    data_y= np.load(r"C:\Users\Dhruv\Downloads\Compressed\RealWorld\realworld_30hz_w10\Y.npy")
    data_p = np.load(r"C:\Users\Dhruv\Downloads\Compressed\RealWorld\realworld_30hz_w10\pid.npy")

    
    le = LabelEncoder()
    le.fit(np.unique(data_y))
    data_y = le.transform(data_y)
    # data_x = np.load(r"C:\Users\Dhruv\Downloads\Compressed\data\data\downstream\realworld_30hz_clean\X.npy")
    # data_y = np.load(r"C:\Users\Dhruv\Downloads\Compressed\data\data\downstream\realworld_30hz_clean\Y.npy")
    # data_p = np.load(r"C:\Users\Dhruv\Downloads\Compressed\data\data\downstream\realworld_30hz_clean\pid.npy")
     

    scenario1_step_subjects = [personid for personid in np.unique(data_p)]
    initial_step_classes = np.unique(data_y)[0:6]
    incremental_step_classes = np.unique(data_y)[6:]
    # initial_step_classes = ['climbingup','climbingdown',
    #                         'jumping','running','walking']
    # incremental_step_classes = ['lying', 'sitting', 'standing']


if(dataset == 'oppo'):
    data_x = np.load(r"C:\Users\Dhruv\Downloads\opportunity+activity+recognition\OpportunityUCIDataset\data\X.npy")
    data_y= np.load(r"C:\Users\Dhruv\Downloads\opportunity+activity+recognition\OpportunityUCIDataset\data\Y.npy")
    data_p = np.load(r"C:\Users\Dhruv\Downloads\opportunity+activity+recognition\OpportunityUCIDataset\data\pid.npy")
    scenario1_step_subjects = [personid for personid in np.unique(data_p)]
    incremental_step_classes = ['406520', '404505', '404508', '404511']
    initial_step_classes = [y for y in np.unique(data_y) if y not in  incremental_step_classes]
    

if(dataset == 'pamap'):
    data_x = np.load(r"C:\Users\Dhruv\Downloads\Compressed\pamap2+physical+activity+monitoring\PAMAP2_Dataset\PAMAP2_Dataset\new_X.npy")
    data_y= np.load(r"C:\Users\Dhruv\Downloads\Compressed\pamap2+physical+activity+monitoring\PAMAP2_Dataset\PAMAP2_Dataset\new_Y.npy")
    data_p = np.load(r"C:\Users\Dhruv\Downloads\Compressed\pamap2+physical+activity+monitoring\PAMAP2_Dataset\PAMAP2_Dataset\new_pid.npy")
    scenario1_step_subjects = [personid for personid in np.unique(data_p)]  # all the subjects
    
    incremental_step_classes = [10 ,18 ,19, 20, 11, 9]
    initial_step_classes = np.array([y for y in np.unique(data_y) if y not in  incremental_step_classes])
    
    print("Classes before transforming", np.unique(data_y))
    le = LabelEncoder()
    le.fit(np.unique(data_y))
    data_y = np.array(le.transform(data_y))
    

    initial_step_classes = le.transform(initial_step_classes)
    incremental_step_classes = le.transform(incremental_step_classes)

    
    classes = np.concatenate([initial_step_classes , incremental_step_classes])

    changed_data_y = []  # Changing the order of the classes
    for i,clas in enumerate(classes):
        data_y[data_y == clas] = -i
    data_y = -data_y
    
    initial_step_classes =np.unique(data_y)[:len(initial_step_classes)]
    incremental_step_classes = np.unique(data_y)[len(initial_step_classes):]
    

    
    

    
    

# Scenario 1
    

scenario =1



print("Number of Classes: {}\nAll Classes:  \t{}".format(len(np.unique(data_y)), np.unique(data_y)))
print("\nIniital Step Classes: {}\nIncremental Step New Classes: {}".format(initial_step_classes, incremental_step_classes))
print('Subjects: ',scenario1_step_subjects)
#

data_initial_step_scenario_1_x = []
data_initial_step_scenario_1_y = []
data_initial_step_scenario_1_p = []

data_incremental_step_scenario_1_x = []
data_incremental_step_scenario_1_y = []
data_incremental_step_scenario_1_p = []


for index, person in enumerate(data_p):
    if(person in scenario1_step_subjects and data_y[index] in initial_step_classes):  # If Person and Class Matches
        data_initial_step_scenario_1_x.append(data_x[index])
        data_initial_step_scenario_1_y.append(data_y[index])
        data_initial_step_scenario_1_p.append(data_p[index])

    elif(person in scenario1_step_subjects and data_y[index] in incremental_step_classes):
        data_incremental_step_scenario_1_x.append(data_x[index])
        data_incremental_step_scenario_1_y.append(data_y[index])
        data_incremental_step_scenario_1_p.append(data_p[index])



# Just randomly shuffling 
random_indices_scenario1 = np.random.permutation(len(data_initial_step_scenario_1_x))  #Initial Step Shuffling
data_initial_step_scenario_1_x, data_initial_step_scenario_1_y, data_initial_step_scenario_1_p =  np.array(data_initial_step_scenario_1_x)[random_indices_scenario1], np.array(data_initial_step_scenario_1_y)[random_indices_scenario1], np.array(data_initial_step_scenario_1_p)[random_indices_scenario1]

random_indices_incremetal_scenario1 = np.random.permutation(len(data_incremental_step_scenario_1_x))  #Incremental Step Shuffling
data_incremental_step_scenario_1_x, data_incremental_step_scenario_1_y, data_incremental_step_scenario_1_p =  np.array(data_incremental_step_scenario_1_x)[random_indices_incremetal_scenario1], np.array(data_incremental_step_scenario_1_y)[random_indices_incremetal_scenario1], np.array(data_incremental_step_scenario_1_p)[random_indices_incremetal_scenario1]


# Initial Step , Here 72% of Old Classes Data is taken for training, 18% for Testing, and Remaining 10% is reserved for Incremental Step
train_len= int(0.72*len(data_initial_step_scenario_1_x))
test_len = train_len + int(0.18*len(data_initial_step_scenario_1_x))  

data_initial_step_scenario_1_x_train, data_initial_step_scenario_1_x_test  = data_initial_step_scenario_1_x[0:train_len], data_initial_step_scenario_1_x[train_len: test_len]
data_initial_step_scenario_1_y_train, data_initial_step_scenario_1_y_test  = data_initial_step_scenario_1_y[0:train_len], data_initial_step_scenario_1_y[train_len: test_len]
data_initial_step_scenario_1_p_train, data_initial_step_scenario_1_p_test = data_initial_step_scenario_1_p[0:train_len], data_initial_step_scenario_1_p[train_len: test_len]

# Getting Validation Data from the Training Data
val_split_idx = int(len(data_initial_step_scenario_1_x_train) * 0.9)  # Taking 10% from the 72% training data for validation
data_initial_step_scenario_1_x_train, data_initial_step_scenario_1_x_val = data_initial_step_scenario_1_x_train[:val_split_idx], data_initial_step_scenario_1_x_train[val_split_idx:]
data_initial_step_scenario_1_y_train, data_initial_step_scenario_1_y_val = data_initial_step_scenario_1_y_train[:val_split_idx], data_initial_step_scenario_1_y_train[val_split_idx:]
data_initial_step_scenario_1_p_train, data_initial_step_scenario_1_p_val = data_initial_step_scenario_1_p_train[:val_split_idx], data_initial_step_scenario_1_p_train[val_split_idx:]


                                             # Data handling for Incremental Stage
# Old Classes in Incremental Stage
train_increment_len =  test_len + int(0.08*len(data_initial_step_scenario_1_y)) # Taking 8% of old Class for Training and 2% for Testing in the Incemental Stage

data_increment_step_scenario_1_x_old_train, data_increment_step_scenario_1_x_old_test  = data_initial_step_scenario_1_x[test_len:train_increment_len], data_initial_step_scenario_1_x[train_increment_len: ]
data_increment_step_scenario_1_y_old_train, data_increment_step_scenario_1_y_old_test  = data_initial_step_scenario_1_y[test_len:train_increment_len], data_initial_step_scenario_1_y[train_increment_len: ]
data_increment_step_scenario_1_p_old_train, data_increment_step_scenario_1_p_old_test  = data_initial_step_scenario_1_p[test_len:train_increment_len], data_initial_step_scenario_1_p[train_increment_len: ]

val_split_idx = int(len(data_increment_step_scenario_1_x_old_train) * 0.9)  # Taking 10% of Training Data of Old Class in Incremental Stage for Validation.
data_increment_step_scenario_1_x_old_train, data_increment_step_scenario_1_x_old_val = data_increment_step_scenario_1_x_old_train[:val_split_idx], data_increment_step_scenario_1_x_old_train[val_split_idx:]
data_increment_step_scenario_1_y_old_train, data_increment_step_scenario_1_y_old_val = data_increment_step_scenario_1_y_old_train[:val_split_idx], data_increment_step_scenario_1_y_old_train[val_split_idx:]
data_increment_step_scenario_1_p_old_train, data_increment_step_scenario_1_p_old_val = data_increment_step_scenario_1_p_old_train[:val_split_idx], data_increment_step_scenario_1_p_old_train[val_split_idx:]


# New Classes in Incremental Stage
train_len= int(0.90*len(data_incremental_step_scenario_1_x))
data_increment_step_scenario_1_x_new_train, data_increment_step_scenario_1_x_new_test  = data_incremental_step_scenario_1_x[0:train_len], data_incremental_step_scenario_1_x[train_len:]
data_increment_step_scenario_1_y_new_train, data_increment_step_scenario_1_y_new_test  = data_incremental_step_scenario_1_y[0:train_len], data_incremental_step_scenario_1_y[train_len:]
data_increment_step_scenario_1_p_new_train, data_increment_step_scenario_1_p_new_test  = data_incremental_step_scenario_1_p[0:train_len], data_incremental_step_scenario_1_p[train_len:]

val_split_idx = int(len(data_increment_step_scenario_1_x_new_train)*0.9)
data_increment_step_scenario_1_x_new_train, data_increment_step_scenario_1_x_new_val = data_increment_step_scenario_1_x_new_train[:val_split_idx], data_increment_step_scenario_1_x_new_train[val_split_idx:]
data_increment_step_scenario_1_y_new_train, data_increment_step_scenario_1_y_new_val = data_increment_step_scenario_1_y_new_train[:val_split_idx], data_increment_step_scenario_1_y_new_train[val_split_idx:]
data_increment_step_scenario_1_p_new_train, data_increment_step_scenario_1_p_new_val = data_increment_step_scenario_1_p_new_train[:val_split_idx], data_increment_step_scenario_1_p_new_train[val_split_idx:]


# Merging Old Classes and new Classses
data_increment_step_scenario_1_x_train = np.concatenate([data_increment_step_scenario_1_x_old_train, data_increment_step_scenario_1_x_new_train])
data_increment_step_scenario_1_y_train = np.concatenate([data_increment_step_scenario_1_y_old_train, data_increment_step_scenario_1_y_new_train])
data_increment_step_scenario_1_p_train = np.concatenate([data_increment_step_scenario_1_p_old_train, data_increment_step_scenario_1_p_new_train])

data_increment_step_scenario_1_x_val = np.concatenate([data_increment_step_scenario_1_x_old_val, data_increment_step_scenario_1_x_new_val])
data_increment_step_scenario_1_y_val = np.concatenate([data_increment_step_scenario_1_y_old_val, data_increment_step_scenario_1_y_new_val])
data_increment_step_scenario_1_p_val = np.concatenate([data_increment_step_scenario_1_p_old_val, data_increment_step_scenario_1_p_new_val])

data_increment_step_scenario_1_x_test = np.concatenate([data_increment_step_scenario_1_x_old_test, data_increment_step_scenario_1_x_new_test])
data_increment_step_scenario_1_y_test = np.concatenate([data_increment_step_scenario_1_y_old_test, data_increment_step_scenario_1_y_new_test])
data_increment_step_scenario_1_p_test = np.concatenate([data_increment_step_scenario_1_p_old_test, data_increment_step_scenario_1_p_new_test])


print("\n                  Train  Val     Test")
print("Initial X \t ", len(data_initial_step_scenario_1_x_train), '\t',len(data_initial_step_scenario_1_x_val), '\t', len(data_initial_step_scenario_1_x_test))
print("Incremental X \t ",len(data_increment_step_scenario_1_x_train),'\t', len(data_increment_step_scenario_1_x_val), '\t',len(data_increment_step_scenario_1_x_test))

print("Initial Y \t ",len(np.unique(data_initial_step_scenario_1_y_train)),'\t', len(np.unique(data_initial_step_scenario_1_y_val)), '\t',len(np.unique(data_initial_step_scenario_1_y_test)))
print("Incremental Y \t ",len(np.unique(data_increment_step_scenario_1_y_train)),'\t', len(np.unique(data_increment_step_scenario_1_y_val)),'\t', len(np.unique(data_increment_step_scenario_1_y_test)))

print("Initial P \t ",len(np.unique(data_initial_step_scenario_1_p_train)),'\t', len(np.unique(data_initial_step_scenario_1_p_val)),'\t', len(np.unique(data_initial_step_scenario_1_p_test)))
print("Incremental P \t ",len(np.unique(data_increment_step_scenario_1_p_train)), '\t',len(np.unique(data_increment_step_scenario_1_p_val)),'\t', len(np.unique(data_increment_step_scenario_1_p_test)))

print("\nClasses in Initial Step: {}\nClasses in Incremental Step: {}".format(np.unique(data_initial_step_scenario_1_y_train), np.unique(data_increment_step_scenario_1_y_train)))
# If error comes run the code 2 times

my_dict_copy = globals().copy()

import os
windowlen = data_initial_step_scenario_1_x_train.shape[1]
os.makedirs('HAR_data/'+ dataset,exist_ok=True)
for name, value in my_dict_copy.items():
    if(name.split("_")[-1] in ['train', 'test','val'] and len(name.split("_"))==7):
        # name = name +".npy"
        np.save('HAR_data/' + dataset + '/'+ name + '_windowLen_' + str(windowlen)+'.npy',value)

print("Shape of Data", data_initial_step_scenario_1_x_train.shape)