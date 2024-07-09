import numpy as np
# def group_items(tot_pop, tot_pop_num, num_groups):
#     # 对物品按照流行度从高到低排序
#     sorted_indices = np.argsort(tot_pop)[:5]
#     sorted_tot_pop = np.array(tot_pop)[sorted_indices]
#
#     # 计算每组的目标流行度之和
#     target_group_pop = tot_pop_num / num_groups
#
#     # 初始化变量
#     item_groups = [0]*10
#     group_pop = 0.0
#     gid = 1;
#     # 遍历排序后的物品
#     for item,pop in enumerate(sorted_tot_pop):
#         if group_pop + pop <= target_group_pop:
#             group_pop += pop
#             item_groups[item] = gid
#         else:
#
#             group_pop = pop
#             gid = gid + 1
#             item_groups[item] = gid
#     print(1)
#
#
#
# # 示例用法
# tot_pop = [0.2, 0.3, 0.1, 0.25, 0.15, 0.08, 0.12, 0.18, 0.05, 0.22]
# tot_pop_num = sum(tot_pop)
# num_groups = 10
#
# group_items(tot_pop, tot_pop_num, num_groups)
# 假设 item_groups 是包含每个物品所属组的数组
item_groups = [1, 2, 1, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
# 假设 top_indices 是前20个推荐物品的下标数组
top_indices = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19])

# 初始化每个组的推荐次数
group_recommendation_count = {group: {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0} for group in set(item_groups)}

# 统计每个组在前1、5、10、20的推荐中出现的次数
for i, item_index in enumerate(top_indices):
    group = item_groups[item_index]
    if i < 1:
        group_recommendation_count[group]['top1'] += 1
    if i < 5:
        group_recommendation_count[group]['top5'] += 1
    if i < 10:
        group_recommendation_count[group]['top10'] += 1
    if i < 20:
        group_recommendation_count[group]['top20'] += 1
a=1
# 输出结果
for group, count_dict in group_recommendation_count.items():
    print(f"Group {group}: Top1={count_dict['top1']}, Top5={count_dict['top5']}, Top10={count_dict['top10']}, Top20={count_dict['top20']}")