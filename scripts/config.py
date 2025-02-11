model_name="VAGOsolutions/SauerkrautLM-Nemo-12b-Instruct-awq"
base_url="https://api.parrotpark.correlaid.org"

num_iter = 2
num_rephrase = 2
beam_size = 4
num_comp = 4
num_questions = 4

chatbot_name = "Bot Botsen"

benchmark_obj_list = [
                # ("arc", 1),
                #   ("mmlu", 1),
                #   ("hellaswag", 1),
                #   ("truthfulqa", 1),
                # weird 404 errors for # encoding 
                #   ("socket_bragging#brag_achievement", 1),
                #   ("socket_hahackathon#is_humor", 1),
                #   ("socket_tweet_irony", 1),
                #   ("socket_sexyn", 1),
                #   ("socket_tweet_offensive", 1),
                #   ("socket_complaints", 1),
                #   ("socket_empathy#empathy_bin", 1),
                #   ("socket_stanfordpoliteness", 1),
                #   ("socket_rumor#rumor_bool", 1),
                #   ("hitom", 1),
                #   ("edos_taska", 1),
                ("ifeval", 1),
                #   ("bbh_boolean_expressions", 1),
                #   ("bbh_causal_judgement", 1),
                #   ("bbh_date_understanding", 1),
                #   ("bbh_disambiguation_qa", 1),
                #   ("bbh_dyck_languages", 1),
                #   ("bbh_formal_fallacies", 1),
                #   ("bbh_geometric_shapes", 1),
                #   ("bbh_hyperbaton", 1),
                #   ("bbh_logical_deduction_five_objects", 1),
                #   ("bbh_logical_deduction_seven_objects", 1),
                #   ("bbh_logical_deduction_three_objects", 1),
                #   ("bbh_movie_recommendation", 1),
                #   ("bbh_multistep_arithmetic_two", 1),
                #   ("bbh_navigate", 1),
                #   ("bbh_object_counting", 1),
                #   ("bbh_penguins_in_a_table", 1),
                #   ("bbh_reasoning_about_colored_objects", 1),
                #   ("bbh_ruin_names", 1),
                #   ("bbh_snarks", 1),
                #   ("bbh_sports_understanding", 1),
                #   ("bbh_temporal_sequences", 1),
                #   ("bbh_tracking_shuffled_objects_five_objects", 1),
                #   ("bbh_tracking_shuffled_objects_seven_objects", 1),
                #   ("bbh_tracking_shuffled_objects_three_objects", 1),
                #   ("bbh_web_of_lies", 1),
                #   ("bbh_word_sorting", 1),
                ]

