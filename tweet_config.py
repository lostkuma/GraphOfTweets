import logging 
import numpy as np
import json 

import queue
import threading 
import multiprocessing 

from heapq import heappush, heapreplace
from typing import Union

# TokenNode, TweetNode, TweetEdge, TweetGraph

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 

# keep top # of values among all the values with a heap	 
class TopValuesMap(object):
	def __init__(self, size:int):
		assert size < 1001, 'can not keep more than 1000 top values'
		self.size = size
		self.top_values = list()
	
	def AddTopValues(self, key, value):
		if self.GetMapSize() < self.size:
			heappush(self.top_values, (value, key))
		else:
			if self.GetMinValue() < value:
				heapreplace(self.top_values, (value, key))
			
	def GetMapSize(self):
		return len(self.top_values)
	
	def GetMinValue(self):
		return self.top_values[0][0]
		
	def GetKeysAndValues(self):
		for value, key in self.top_values:
			yield key, value 


# define graph nodes
class TokenNode(object):
	def __init__(self, token:str):
		self.token = token
		self.token_freq = 0
		self.tokens_merged_into_this_node = [] # {token1, token2, ..}

	def AddMergedToken(self, merged_token:str):
		self.tokens_merged_into_this_node.append(merged_token)

	def GetMergedTokens(self):
		return self.tokens_merged_into_this_node

	def GetNumMergedTokens(self, including_self=False):
		if including_self:
			return len(self.tokens_merged_into_this_node) + 1
		return len(self.tokens_merged_into_this_node)

	def GetToken(self):
		return self.token

	def GetTokenFreq(self):
		return self.token_freq
	
	def UpdateTokenFreq(self, value=0):
		self.token_freq += value


# define tweet graph nodes
class TweetNode(object):
	def __init__(self, index:int, tweet:Union[set, list], leading_tokens:Union[set, list, None]=None, intersection_sum=0):
		self.index = index # int 
		self.tweet = tweet # original processed tweet: set of words
		# self.date = None # 4-digit string i.e.'0225'
		# self.intersection_sum = intersection_sum # int, sum of all sizes of intersections between other tweets 
		
		self.tweet_merged_into_this_node = set() # {tweetidx1, tweetidx2, } 
		
		if not leading_tokens:
			self.leading_tokens_in_tweet = set() # {leading_token_of_merged_node1, }
		else:
			self.leading_tokens_in_tweet = set(leading_tokens)
			
		self.idx_neighbor_to_edge = dict() # {neighbor_tweet_idx1: tweetedge, }

	def AddLeadingToken(self, leading_token:str):
		self.leading_tokens_in_tweet.add(leading_token)
		
	def UpdateLeadingTokens(self, leading_tokens:set):
		self.leading_tokens_in_tweet = leading_tokens 

	def GetLeadingTokens(self):
		return self.leading_tokens_in_tweet

	def GetTweetText(self):
		return self.tweet
		
	def UpdateTweetText(self, tweet:set):
		self.tweet = tweet 

	def GetTweetIndex(self):
		return self.index 

	def GetNumOfLeadingTokens(self):
		return len(self.leading_tokens_in_tweet)

	def GetTweetLength(self):
		return len(self.tweet)

	def SetDate(self, date:str):
		self.date = date 

	def GetTweetDate(self):
		if self.date == None: 
			print('no date data available! returning None')
		return self.date 
		
	def UpdateIntersectionSum(self, value=0):
		self.intersection_sum += value 
	
	def GetIntersectionSum(self):
		return self.intersection_sum 
		
	def ResetIntersectionSum(self):
		self.intersection_sum = 0 

	def AddEdge(self, edge):
		node_a = edge.GetNodeA()
		node_b = edge.GetNodeB()
		neighbor_token = None
		if node_a.GetTweetIndex() == self.GetTweetIndex():
			neighbor_idx = node_b.GetTweetIndex()
		else:
			neighbor_idx = node_a.GetTweetIndex()
		self.idx_neighbor_to_edge[neighbor_idx] = edge
	
	def GetTotalEdgeWeight(self):
		total_weight = None 
		for _, edge in self.idx_neighbor_to_edge.items():
			if not total_weight: 
				total_weight = edge.GetWeight()
			else:
				total_weight += edge.GetWeight()
		return total_weight 
	
	def GetWeightedNeighbors(self):
		# generate weight(int), neighbor_tweet(TweetNode) pairs 
		for _, edge in self.idx_neighbor_to_edge.items():
			node_a = edge.GetNodeA()
			node_b = edge.GetNodeB()
			neighbor = None 
			if node_a.GetTweetIndex() == self.GetTweetIndex():
				neighbor = node_b
			else:
				neighbor = node_a 
			yield edge.GetWeight(), neighbor 
			
	def GetIntersection(self, node):
		tokens1 = self.GetLeadingTokens()
		tokens2 = node.GetLeadingTokens()
		return tokens1.intersection(tokens2) 
		
	def HasEdge(self, idx): # if another node has edge with current node 
		return idx in self.idx_neighbor_to_edge
		
	def GetMergedTweetIndices(self):
		return self.tweet_merged_into_this_node
		
	def AddMergedTweetIndex(self, idx):
		self.tweet_merged_into_this_node.add(idx)
	

# define tweet graph edges
class TweetEdge(object):
	def __init__(self, node_a, node_b, weight):
		self.node_a = node_a
		self.node_b = node_b
		self.weight = weight

	def GetNodeA(self):
		return self.node_a

	def GetNodeB(self):
		return self.node_b

	def GetWeight(self):
		return self.weight

	def SetWeight(self, new_weight):
		self.weight = new_weight


class TweetGraph(object):
	def __init__(self):
		self.token_to_node = dict() # {token: TokenNode }
		self.token_to_merged_token = dict() # {already_merged_token: merged_into_token}
		# self.token_pair_to_PMI = dict() # {token_pair ('min,max'): pmi between tokens}
		self.tweet_idx_to_node = dict() #{tweet_idx: TweetNode }
		# self.mi_denominator = 0
		# self.token_to_tweet_mapping = None 
		self.tweet_idx_to_merged_tweet = dict() # {merged_idx: tweet_idx, }

	def GetNumOfNodes(self):
		return len(self.token_to_node)

	def GetAllTokens(self):
		return list(self.token_to_node.keys())

	def HasTokenNode(self, token:str):
		return token in self.token_to_node

	def GetTokenNode(self, token:str):
		if token in self.token_to_node:
			return self.token_to_node[token]
		if token in self.token_to_merged_token: 
			# if token was merged already, return mapped token
			return self.GetTokenNode(self.token_to_merged_token[token])
		assert False, 'Token not in graph: ' + token

	def AddToken(self, token:str):
		node = TokenNode(token)
		self.token_to_node[token] = node

	def GetNumOfTweets(self):
		return len(self.tweet_idx_to_node)
	
	def HasTweet(self, idx:int):
		return idx in self.tweet_idx_to_node
		
	def AddTweet(self, idx:int, tweet_text:set, leading_tokens=None, intersection_sum=0):
		assert idx not in self.tweet_idx_to_node, 'trying to add an existing index! use UpdateTweet() for modifications!'
		node = TweetNode(idx, tweet_text, leading_tokens, intersection_sum)
		self.tweet_idx_to_node[idx] = node 

	def UpdateTweet(self, idx:int, tweet_text=None, leading_tokens=None, intersection_sum=None):
		if tweet_text:
			self.GetTweetNode(idx).UpdateTweetText(tweet_text)
		if leading_tokens:
			self.GetTweetNode(idx).UpdateLeadingTokens(leading_tokens)
		if intersection_sum:
			self.GetTweetNode(idx).UpdateIntersectionSum(intersection_sum)
		
	def GetTweetNode(self, idx):
		if idx in self.tweet_idx_to_node:
			return self.tweet_idx_to_node[idx]
			
		# if the idx was merged already, return mapped node 
		if idx in self.tweet_idx_to_merged_tweet: 
			return self.GetTweetNode(self.tweet_idx_to_merged_tweet[idx])
		assert False, 'Tweet index not in graph: ' + str(idx)
		
	def AddMergedTweet(self, idx, idx_to_merge): 
		self.tweet_idx_to_merged_tweet[idx_to_merge] = idx
		
	def AddTweetEdge(self, idx1, idx2, weight):
		assert idx1 != idx2, 'trying to add self edge ' + idx1
		node1 = self.GetTweetNode(idx1)
		node2 = self.GetTweetNode(idx2)
		if not node1.HasEdge(idx2):
			edge = TweetEdge(node1, node2, weight)
			node1.AddEdge(edge)
			node2.AddEdge(edge)

	def GetAllTweetIdx(self): 
		return list(self.tweet_idx_to_node.keys()) 

	def MakeTokenToTweetMapping(self):
		# make token to tweet mapping {token: [tweet_idx, ], }
		self.token_to_tweet_mapping = dict(zip(self.GetAllTokens(), [[] for i in range(self.GetNumOfNodes())]))

		for idx in self.GetAllTweetIdx():
			tokens_in_tweet = self.GetTweetNode(idx).GetLeadingTokens()
			for token in tokens_in_tweet:
				self.token_to_tweet_mapping[token].append(idx)
				
	def GetTokenToTweetMapping(self):
		assert self.token_to_tweet_mapping != None, 'the mapping does not exit! make it with MakeTokenToTweetMapping() first'
		
		for token, tweets_idx in self.token_to_tweet_mapping.items():
			yield token, tweets_idx 

	def GetSortedTokenFreq(self, reverse=False):
		token_freqs = dict(zip(self.GetAllTokens(), [0 for i in range(self.GetNumOfNodes())]))

		for token, idxs in self.GetTokenToTweetMapping():
			freq = len(idxs)
			token_freqs[token] = freq

		token_freqs = sorted(token_freqs.items(), key=lambda x: x[1], reverse=reverse)
		for token, freq in token_freqs:
			yield token, freq 

	def GetMI(self, idx1:int, idx2:int, normalize:bool=True):
		# p_x; p_y = num unique token node in tweet over num all tokens 
		# p_xy = intersection size of two tweet over num all tokens 
		# MI = log(p_xy / (p_x * p_y))
		assert idx1 in self.tweet_idx_to_node and idx2 in self.tweet_idx_to_node, 'Index not in graph!'

		tweet1 = self.GetTweetNode(idx1)
		tweet2 = self.GetTweetNode(idx2)

		intersection_size = len(tweet1.GetIntersection(tweet2))

		mi_denominator = self.GetNumOfNodes() 
		tweet1_size = tweet1.GetNumOfLeadingTokens()
		tweet2_size = tweet2.GetNumOfLeadingTokens()

		# if no intersection (negative infinity), return -1
		if intersection_size == 0:
			return -1

		p_1 = tweet1_size / mi_denominator
		p_2 = tweet2_size / mi_denominator
		p_intersection = intersection_size / mi_denominator

		mi = np.log(p_intersection / (p_1 * p_2))

		# use max(-log(p1), -log(p2)) to normalize the pointwise tweet MI 
		# normalize to [-1, 1]
		if normalize:
			norm12 = (-1) * np.min([np.log(p_1), np.log(p_2)]) 
			mi = mi / norm12 
		
		return mi 

	def GetAllNeighboringMIs(self, target_idx:int, normalize:bool=True):
		all_indices = self.GetAllTweetIdx()
		for idx in all_indices:
			if idx != target_idx:
				mi = self.GetMI(target_idx, idx, normalize)
				yield idx, mi 

	def MakeIdxKey(self, idx1, idx2):
		assert idx1 != idx2, 'trying to make self index key'
		if idx1 < idx2:
			return str(idx1) + ',' + str(idx2)
		else:
			return str(idx2) + ',' + str(idx1)

	def GetIdxKeyAndMIThread(self, target_idx, normalize:bool=True, filter_size:int=2):
		# given a tweet idx, compute MI between this idx and all idx that larger than it 
		# use intersection criteria to filter some MIs that will not be ranked in top MI
		# filter_size: i.e. if intersection is only one tweet_node (filter_size=1), then skip the pairs 
		# thread worker function 
		all_indices = self.GetAllTweetIdx() 
		
		for idx in all_indices: 
			if target_idx < idx: 

				tweet1 = self.GetTweetNode(target_idx)
				tweet2 = self.GetTweetNode(idx)
				intersection_size = len(tweet1.GetIntersection(tweet2))
				
				if intersection_size >= filter_size:
					idx_key = self.MakeIdxKey(target_idx, idx)
					mi = self.GetMI(target_idx, idx, normalize)
					
					yield idx_key, mi 


	def GetTopMIs(self, size:int, normalize:bool=True, filter_size:int=2, num_threads:int=multiprocessing.cpu_count()):
		# get top_n MIs and pair indices from the graph, only used for when topn < a size 
		# keep two ordered arrays, one with idx_key, one with MI values
		# pop the correct indices every time to keep track of top_n 
		
		assert size < 1001, 'keep size reasonable! '
		
		tweet_id_work_queue = queue.Queue() 
		tweet_mi_output_queue = queue.Queue() 
		
		top_values_final_dict = dict() 
		
		all_indices = self.GetAllTweetIdx()
		all_indices = sorted(all_indices) 
		
		def ThreadWorkerFunction(thread_id, size:int):
			top_heap = TopValuesMap(size)
			
			#print('thread ' + str(thread_id) + ' started')
			
			while True: 
				target_idx = tweet_id_work_queue.get() 

				if target_idx == None:
					break 

				if target_idx % 500 == 0:
					logging.info('thread ' + str(thread_id) + ' working on tweet ' + str(target_idx))
				
				# compute the MI values to one tweet index and add them to top heap 
				for idx_key, mi in self.GetIdxKeyAndMIThread(target_idx, normalize, filter_size): 
					top_heap.AddTopValues(idx_key, mi)

			tweet_mi_output_queue.put(top_heap)

		logging.info('threading started...')
		
		# add each tweet idx once to the queue 
		for idx in all_indices: 
			tweet_id_work_queue.put(idx)
			
		# add None for each thread to mark the end of each thread 
		for _ in range(num_threads): 
			tweet_id_work_queue.put(None)
			
		threads = [threading.Thread(target=ThreadWorkerFunction, args=[thread_id, size], daemon=True) 
					for thread_id in range(num_threads)]
		
		for t in threads: 
			t.start() 
		for t in threads: 
			t.join() 
		
		logging.info('threading completed, combining output...')
		
		top_values_final = TopValuesMap(size) 
		
		for i in range(num_threads):
			top_value_thread = tweet_mi_output_queue.get()
			for key, val in top_value_thread.GetKeysAndValues():
				top_values_final.AddTopValues(key, val) 
		
		for key, val in top_values_final.GetKeysAndValues():
			top_values_final_dict[key] = val 
			
		return top_values_final_dict 

	def IdentifyMergeTweetsThread(self, all_indices, target_idx:int):
		# identify what tweets are identical and needs to be merged 
		# always check the target_idx with the larger indices 
		# output a set of tweet index pairs that needs to be merged 
		# thread worker function 
		
		for idx in all_indices: 
			# check all indices that's bigger than the target index 
			if target_idx < idx: 
				tweet1 = self.GetTweetNode(target_idx)
				tweet2 = self.GetTweetNode(idx) 
				
				leading_tokens1 = tweet1.GetLeadingTokens()
				leading_tokens2 = tweet2.GetLeadingTokens() 
				
				if leading_tokens1 == leading_tokens2: 
					# separate the tweet index with a comma, keep the smaller one in front 
					index_pair = str(target_idx) + ',' + str(idx)
					yield index_pair 

	def Merge_TweetB_Into_TweetA(self, idx1, idx2):
		# should be used when tweet1 and tweet2 are identical tweet but different objects
		# tweet2 will be removed from the graph and added as an index into tweet1 
		
		##################################
		# check if tweet2 is already removed, remove tweet2 from the graph
		# note the set used for generating pairs of indices can repeat and are not ordered (when tweet3 is merged into tweet1, tweet1 and tweet2 are identical, and we're merging tweet3 into tweet2 case)
		
		tweet_a = self.GetTweetNode(idx1) 
		tweet_b = self.GetTweetNode(idx2) 
		
		# if tweet_a and tweet_b are identical tweet, no need to merge 
		if tweet_a == tweet_b: 
			logging.info('\t[already merged] skip merging tweets ' + str(idx1) + ' and ' + str(idx2))
			return 
		
		# if the original tweet of idx2 has not been merged, merge idx2 
		# if idx2 has already been merged into other tweet, merge that tweet and migrate the merged indices 
		idx_a = tweet_a.GetTweetIndex() 
		idx_b = tweet_b.GetTweetIndex()

		# add idx_b as tweet_a's merged index 
		tweet_a.AddMergedTweetIndex(idx_b) 

		# check if the tweet_b already has merged tweet in it 
		indices_already_merged = tweet_b.GetMergedTweetIndices() 
		for idx_merged in indices_already_merged:
			tweet_a.AddMergedTweetIndex(idx_merged)

		# add tweet_b into the merged index to tweet lookup dictionary in the graph 
		self.AddMergedTweet(idx_a, idx_b)	

		# remove the merged tweet node from the graph 
		self.tweet_idx_to_node.pop(idx_b)


	def MergeIdenticalTweets(self, num_threads:int=multiprocessing.cpu_count()):
		# merge tweets that have the same leadning tokens into one node 
		
		tweet_id_work_queue = queue.Queue() 
		identical_tweet_pair_queue = queue.Queue() 
		
		# keep a copy of all indices so we are not iterating while deleting indices 
		all_indices = self.GetAllTweetIdx() 
		all_indices = sorted(all_indices) 
		
		identical_tweet_pairs_final = set() 

		def ThreadWorkerFunction(thread_id, all_indices):
			identical_tweet_pairs = set() 
			
			while True: 
				target_idx = tweet_id_work_queue.get() 

				if target_idx == None:
					break 

				if target_idx % 500 == 0:
					logging.info('thread ' + str(thread_id) + ' working on tweet ' + str(target_idx))
				
				# identify pairs needs to be merged for every index 
				for index_pair in self.IdentifyMergeTweetsThread(all_indices, target_idx):
					identical_tweet_pairs.add(index_pair)
			
			identical_tweet_pair_queue.put(identical_tweet_pairs)		

		logging.info('threading started...')

		# add each tweet idx once to the queue 
		for idx in all_indices: 
			tweet_id_work_queue.put(idx)
			
		# add None for each thread to mark the end of each thread 
		for _ in range(num_threads): 
			tweet_id_work_queue.put(None)
			
		threads = [threading.Thread(target=ThreadWorkerFunction, args=[thread_id, all_indices], daemon=True) 
					for thread_id in range(num_threads)]
		
		for t in threads: 
			t.start() 
		for t in threads: 
			t.join() 

		logging.info('threading completed, combining output...')

		# add all identical tweet pairs together 
		for i in range(num_threads):
			identical_tweet_pairs_thread = identical_tweet_pair_queue.get()
			identical_tweet_pairs_final.update(identical_tweet_pairs_thread)

		# merge all identical pairs	 
		num_merges = 0
		
		for pair in identical_tweet_pairs_final:
			pair = pair.split(',')
			idx1 = int(pair[0]) 
			idx2 = int(pair[1]) 
			
			# merge tweet2 into tweet1 
			self.Merge_TweetB_Into_TweetA(idx1, idx2)
			num_merges += 1
		
		logging.info('finished merging identical tweets! ' + str(num_merges) + ' merges completed.')
		

	# ====== different way of computing MI, treat tweets as clusters ======
	
	def GetMIDenominator(self):
		return self.mi_denominator 

	def GetMIAsCluster(self, idx1:int, idx2:int, normalize:bool=True):
		# returns the mi between two tweet idx 
		# mi between tweet as mi between clusters 
		# mi = log(p(x,y) / p(x)p(y))
		assert idx1 in self.tweet_idx_to_node and idx2 in self.tweet_idx_to_node, 'Index not in graph!'

		tweet1 = self.GetTweetNode(idx1)
		tweet2 = self.GetTweetNode(idx2)
		
		intersection_size = len(tweet1.GetIntersection(tweet2))
		
		# if no intersection (negative infinity), return -1
		if intersection_size == 0:
			return -1 

		mi_denominator = self.GetMIDenominator()
		
		p_1 = tweet1.GetIntersectionSum() / mi_denominator
		p_2 = tweet2.GetIntersectionSum() / mi_denominator
		p_intersection = intersection_size / mi_denominator
		
		mi = np.log(p_intersection / (p_1 * p_2))	
		
		# use max(-log(p1), -log(p2)) to normalize the pointwise tweet MI 
		if normalize:
			norm12 = (-1) * np.min([np.log(p_1), np.log(p_2)]) 
			mi = mi / norm12 
		
		return mi 

	def GetAllNeighborMIsAsCluster(self, target_idx:int, normalize:bool=True):
		all_indices = self.GetAllTweetIdx()
		for idx in all_indices:
			if idx != target_idx:
				mi = self.GetMIAsCluster(target_idx, idx, normalize)
				yield idx, mi 

	def InitializeMIsAsCluster(self, normalize:bool=True, output_filename=None):
		# initialize the MI computation between tweets for a new graph 
		# if update only appear at point level then no need to run this function (instead: update with tweet node and graph denominator sum)

		print('initializing MIs...')
		tweet_indices = self.GetAllTweetIdx()
		est_num_pairs = (len(tweet_indices) ** 2) / 2 # estimated number of tweet pairs	 
		progress = 0

		for i in range(len(tweet_indices) - 1): 
			for j in range(i+1, len(tweet_indices)):

				idx1 = tweet_indices[i] 
				idx2 = tweet_indices[j] 
				node1 = self.GetTweetNode(idx1)
				node2 = self.GetTweetNode(idx2) 

				intersection_size = len(node1.GetIntersection(node2))
				
				# if has intersection, update the intersection_size for both nodes and the total_n 
				if not intersection_size == 0:
					node1.UpdateIntersectionSum(intersection_size)
					node2.UpdateIntersectionSum(intersection_size)
					self.UpdateMIDenominator(intersection_size)

				progress += 1
				if progress % int(est_num_pairs *0.1) == 0:
					print('\tprogress for intersections: ' + str(progress / est_num_pairs * 100) + '% of all tweet processed')
		
		if output_filename:
			self.SaveTweetsToJson(output_filename)

		print('MI initialization completed. Use GetMI() to compute MI between tweet idx pairs')


	def LoadTokensFromJson(self, input_filename):
		json_graph = None
		with open(input_filename, 'r', encoding='utf-8') as input_file:
			json_graph = json.load(input_file)

		# adding nodes back
		for node in json_graph["nodes"]:
			node_token = node["token"]
			node_merged_tokens = node["merged_tokens"]

			self.AddToken(node_token)
			node = self.GetTokenNode(node_token)
			for node_merged_token in node_merged_tokens:
				node.AddMergedToken(node_merged_token)
				self.token_to_merged_token[node_merged_token] = node_token

	def SaveTweetsToJson(self, output_filename):
		json_tweets = []
		
		for idx, tweet in self.tweet_idx_to_node.items():
			json_tweet = {
				'idx': idx, 
				'tweet_text': list(tweet.GetTweetText()), 
				'leading_tokens': list(tweet.GetLeadingTokens()),
				'tweet_merged': list(tweet.GetMergedTweetIndices()),
			}
			json_tweets.append(json_tweet)
		
		with open(output_filename, 'w', encoding='utf-8') as output_file:
			json.dump(json_tweets, output_file, ensure_ascii=False, indent=2)

	def LoadTweetsFromJson(self, input_filename):
		json_tweet_graph = None 
		with open(input_filename, 'r', encoding='utf-8') as input_file:
			json_tweet_graph = json.load(input_file) 
		
		# adding tweet nodes back 
		for node in json_tweet_graph:
			tweet_idx = node['idx']
			tweet_text = set(node['tweet_text'])
			leading_tokens = set(node['leading_tokens'])

			self.AddTweet(tweet_idx, tweet_text, leading_tokens)

			for merged_idx in node['tweet_merged']:
				self.AddMergedTweet(tweet_idx, merged_idx) 
				self.GetTweetNode(tweet_idx).AddMergedTweetIndex(merged_idx)
