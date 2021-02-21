python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 1 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_1
python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 2 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_2
python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 3 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_3
python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 4 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_4
python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_gru_5k_with_emb --layer-num 1 --num-epochs 4 --suffix-name _sentiment_gru_with_emb_on_5k_at_layer_1
python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_gru_5k_with_emb --layer-num 2 --num-epochs 4 --suffix-name _sentiment_gru_with_emb_on_5k_at_layer_2
python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_gru_5k_with_emb --layer-num 3 --num-epochs 4 --suffix-name _sentiment_gru_with_emb_on_5k_at_layer_3
python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_gru_5k_with_emb --layer-num 4 --num-epochs 4 --suffix-name _sentiment_gru_with_emb_on_5k_at_layer_4


python train.py probing data/bigram_order_train.jsonl data/bigram_order_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 4 --num-epochs 8 --suffix-name _bigram_order_dan_with_emb_on_5k_at_layer_4
python train.py probing data/bigram_order_train.jsonl data/bigram_order_dev.jsonl --base-model-dir serialization_dirs/main_gru_5k_with_emb --layer-num 4 --num-epochs 4 --suffix-name _bigram_order_gru_with_emb_on_5k_at_layer_4
