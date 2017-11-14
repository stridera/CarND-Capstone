
import os
import sys

import tensorflow as tf
import yaml
from object_detection.utils import dataset_util

from random import shuffle

try:
  OBJECT_DETECTION_API_FOLDER = os.environ['OBJECT_DETECTION_API_FOLDER']
  TRANSFER_LEARNING_FOLDER = os.environ['CURRENT_TRANSFER_LEARNING_FOLDER']
  BOSCH_DATA_FOLDER = os.environ['BOSCH_DATA_FOLDER']
except:
  sys.exit('ERROR: Run set_env.bash to have the folders information as environmental variables.')

CONVERTED_DATA_FOLDER = TRANSFER_LEARNING_FOLDER + '/data'
sys.path.insert(0, OBJECT_DETECTION_API_FOLDER)

flags = tf.app.flags
flags.DEFINE_string('output_path', CONVERTED_DATA_FOLDER, 'Path to output TFRecord')
FLAGS = flags.FLAGS

# IMPORTANT: in the config file set the number of classes folliwing the number of classes set here.
LABEL_DICT =  {
    'Green' : 1,    
    'Red' : 2,  
    'RedLeft' : 3,  
    'off' : 4,  
    'Yellow' : 5,   
    'GreenLeft' : 6,    
    'Occluded' : 7, 
#    'GreenRight' : 8,  
#    'RedStraight' : 9, 
#    'RedRight' : 10,   
#    'GreenStraight' : 11,  
#    'RedStraightLeft' : 12,    
#    'RedStraightRight' : 13,   
#    'GreenStraightRight' : 14, 
#    'GreenStraightLeft' : 15,  
    }

OMIT_LABELS_LIST = [
    'GreenLeft',
    'Occluded',
    'GreenRight',
    'RedStraight',
    'RedRight',
    'GreenStraight',
    'RedStraightLeft',
    'RedStraightRight',
    'GreenStraightRight',
    'GreenStraightLeft',
    ]

def get_dataset_stats(images, output_file = None):
    """
    Prints statistic data for a list of samples from the Bosch dataset.

    :param images: samples list
    :param output_file: If not None, it is the name of the file to store the stats.
    """
    
    num_images = len(images)
    num_lights = 0
    appearances = {'occluded': 0}

    for image in images:
        num_lights += len(image['boxes'])
        for box in image['boxes']:
            try:
                appearances[box['label']] += 1
            except KeyError:
                appearances[box['label']] = 1

            if box['occluded']:
                appearances['occluded'] += 1

    output_text =  'Number of images: ' + str(num_images) + '\n'
    output_text += 'Number of traffic lights: ' + str(num_lights) + '\n'
    
    output_text += 'Labels:\n'
    for k, l in appearances.items():
        output_text += '{: 10} [{: 3} %] {}\n'.format(l, int(l/num_lights*100), k)
        

    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write(output_text)
    else:
        print(output_text)    
        
        
def create_tf_example(example):
    
    # Bosch
    height = 720 # Image height
    width = 1280 # Image width

    filename = example['path'] # Filename of the image. Empty if image is not from file
    filename = filename.encode()

    with tf.gfile.GFile(example['path'], 'rb') as fid:
        encoded_image = fid.read()

    image_format = 'png'.encode() 

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in example['boxes']:
        xmins.append(float(box['x_min'] / width))
        xmaxs.append(float(box['x_max'] / width))
        ymins.append(float(box['y_min'] / height))
        ymaxs.append(float(box['y_max'] / height))
        classes_text.append(box['label'].encode())
        classes.append(int(LABEL_DICT[box['label']]))


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

def generate_tf_record(output_file_name, samples):
    
    get_dataset_stats(samples, CONVERTED_DATA_FOLDER + '/' + output_file_name + '_stats.txt')
    
    output_file_path = FLAGS.output_path + '/' + output_file_name + '.record'
    if os.path.isfile(output_file_path):
        sys.exit('ERROR: The record file exist already. If you want to create a new one please delete or rename the old one.\n' + output_file_path)

    if not os.path.isdir(CONVERTED_DATA_FOLDER):
      os.mkdir(CONVERTED_DATA_FOLDER)

    len_samples = len(samples)
    writer = tf.python_io.TFRecordWriter(output_file_path)
    
    counter = 0
    for sample in samples:
        tf_example = create_tf_example(sample)
        writer.write(tf_example.SerializeToString())
    
        if counter % 10 == 0:
            print('Adding {}, done {:.0f} % ...'.format(output_file_name, (counter/len_samples)*100), end='\r')
        counter += 1
    
    print('\n')
    writer.close()
    

# BOSCH
print('Reading Bosch Yaml file...')
INPUT_YAML = BOSCH_DATA_FOLDER + '/train.yaml'
all_samples = yaml.load(open(INPUT_YAML, 'rb').read())


len_samples = len(all_samples)

# Get paths and remove samples to be omited.
list_to_remove = []
for i in range(len_samples):

    # If the sample contains a traffic light to be ommited we omit it (even if other boxes are ok)
    for box in all_samples[i]['boxes']:
        if box['label'] in OMIT_LABELS_LIST:
            #print('Should be eliminated:', i, box['label'])
            list_to_remove.append(i)
            break
            
    all_samples[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(INPUT_YAML), all_samples[i]['path']))

#del all_samples[list_to_remove]
for num in sorted(list_to_remove, reverse=True):
    #print('Del item', num, ' from list size of: ', len(all_samples))
    del all_samples[num]

len_samples = len(all_samples)
print('Samples remaining: ', len_samples)

print('Loaded ', len_samples, 'samples in total.')

train_fraction = 0.7
len_train_samples = int(train_fraction * len_samples)
len_val_samples = len_samples - len_train_samples

print('Train samples: {}, validation samples: {}'.format(len_train_samples, len_val_samples))

# Shuffle before splitting
reshuffle = True
while(reshuffle):
    shuffle(all_samples)
    train_samples = all_samples[:len_train_samples]
    val_samples = all_samples[:len_val_samples]
    print('Training samples:')
    get_dataset_stats(train_samples)
    print('Validation samples:')
    get_dataset_stats(val_samples)
    
    answer = input('These are the stats of the data, continue? (no - reshuffle, anything else - go)\n')
    if answer.lower() != 'no':
        reshuffle = False
        

generate_tf_record('train_data', train_samples)
generate_tf_record('val_data', val_samples)

# Generate the label map
text = ''
for key, item in LABEL_DICT.items():
  new_text = "item {{\n  id: {}\n  name: '{}'\n}}\n\n".format(item, key)
  text = text + new_text

with open(CONVERTED_DATA_FOLDER + '/bosch_label_map.pbtxt', 'w') as f:
  f.write(text)

