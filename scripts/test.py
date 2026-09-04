from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

print(r1_zero_reward_fn("test123 </think> <answer> 18 </answer> ", "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.\n#### 18"))