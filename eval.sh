echo "evaluate english sts:"
python3 eval/evaluation.py --model_name_or_path $1
echo "evaluate retireval"
python3 eval/evaluation_retrieval.py --model_name_or_path $1
#echo "evaluate classification:"
#python3 classification.py --model_name_or_path $1
echo "evaluate xsts dataset:"
python3 eval/evaluation_xsts.py --model_name_or_path $1
#echo "analysis embedding:"
#python3 analysis_embed.py --model_name_or_path $1
