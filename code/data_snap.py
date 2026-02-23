import tensorflow_datasets as tfds


snap_path = "/path/to/your/dataset.snap"
builder = tfds.builder_from_snapshot(snap_path)
dataset = builder.as_dataset()


import os
import tensorflow as tf

output_dir = "/path/to/split_dataset"
os.makedirs(output_dir, exist_ok=True)

# iterujemy po batchach
for i, batch in enumerate(dataset):
    shard_path = os.path.join(output_dir, f"part_{i}.snap")
    tf.data.experimental.save(batch, shard_path)
