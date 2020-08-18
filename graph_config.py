import json 
import logging

# define graph edges
class GraphEdge(object):	
	
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

	def AddWeight(self, additional_weight):
		self.weight += additional_weight

	def SetWeight(self, new_weight):
		self.weight = new_weight
		
		
# define graph nodes
class GraphNode(object):
	def __init__(self, token):
		self.token = token
		self.token_freq = 0
		self.token_neighbor_to_edge = dict() # {neighbor_token1: node_edge, ..}
		self.tokens_merged_into_this_node = [] # {token1, token2, ..}

	def AddMergedToken(self, merged_token):
		self.tokens_merged_into_this_node.append(merged_token)

	def GetMergedTokens(self):
		return self.tokens_merged_into_this_node

	def GetToken(self):
		return self.token

	def GetEdge(self, token):
		return self.token_neighbor_to_edge[token]

	def GetTokenFreq(self):
		return self.token_freq
	
	def UpdateTokenFreq(self, value):
		self.token_freq += value
	
	def GetTotalEdgeWeight(self):
		total_weight = None
		for _, edge in self.token_neighbor_to_edge.items():
			if not total_weight:
				total_weight = edge.GetWeight()
			else:
				total_weight += edge.GetWeight()
		return total_weight

	def GetWeightedNeighbors(self):
		for _, edge in self.token_neighbor_to_edge.items():
			node_a = edge.GetNodeA()
			node_b = edge.GetNodeB()
			neighbor = None
			if node_a.GetToken() == self.GetToken():
				neighbor = node_b
			else:
				neighbor = node_a
			yield edge.GetWeight(), neighbor

	def GetWeightedNeighborTokens(self):
		for token, edge in self.token_neighbor_to_edge.items():
			yield edge.GetWeight(), token

	def GetNeighborTokens(self):
		for token in self.token_neighbor_to_edge:
			yield token

	def AddEdge(self, edge):
		node_a = edge.GetNodeA()
		node_b = edge.GetNodeB()
		neighbor_token = None
		if node_a.GetToken() == self.GetToken():
			neighbor_token = node_b.GetToken()
		else:
			neighbor_token = node_a.GetToken()
		self.token_neighbor_to_edge[neighbor_token] = edge

	def RemoveEdge(self, neighbor_token):
		if neighbor_token in self.token_neighbor_to_edge:
			self.token_neighbor_to_edge.pop(neighbor_token)

	def HasNeighbor(self, token):
		return token in self.token_neighbor_to_edge

	def HasEdge(self, token): # if another node has edge with current node 
		return token in self.token_neighbor_to_edge
		
		
class Graph(object):
	def __init__(self):
		self.token_to_node = dict()
		self.token_to_merged_token = dict() 
		self.num_merges = 0

	def GetNumOfNodes(self):
		return len(self.token_to_node)

	def GetAllTokens(self):
		return list(self.token_to_node.keys())

	def GetNumMerges(self):
		return self.num_merges

	def HasNode(self, token):
		return token in self.token_to_node

	def GetNode(self, token):
		if token in self.token_to_node:
			return self.token_to_node[token]
		if token in self.token_to_merged_token: # if token was merged already, return mapped token
			return self.GetNode(self.token_to_merged_token[token])
		assert False, 'Token not in graph: ' + token

	def AddNode(self, token):
		node = GraphNode(token)
		self.token_to_node[token] = node

	def AddEdge(self, token_a, token_b, weight):
		assert token_a != token_b, 'trying to add self edge ' + token_a
		node_a = self.GetNode(token_a)
		node_b = self.GetNode(token_b)
		if not node_a.HasNeighbor(token_b):
			edge = GraphEdge(node_a, node_b, weight)
			node_a.AddEdge(edge)
			node_b.AddEdge(edge)

	def AddEdgeOrAccumulateEdgeWeight(self, token_a, token_b, weight):
		assert token_a != token_b, 'trying to add self edge' + token_a
		node_a = self.GetNode(token_a)
		node_b = self.GetNode(token_b)
		if node_a.HasEdge(token_b):
			edge = node_a.GetEdge(token_b)
			edge.AddWeight(weight)
		else:
			edge = GraphEdge(node_a, node_b, weight)
			node_a.AddEdge(edge)
			node_b.AddEdge(edge)

	def RemoveEdge(self, token_a, token_b):
		node_a = self.GetNode(token_a)
		node_b = self.GetNode(token_b)
		node_a.RemoveEdge(token_b)
		node_b.RemoveEdge(token_a)

	def Print(self):
		print('Graph: ')
		for _, node in self.token_to_node.items():
			print('	 Graph Node { token: ', node.GetToken(), 'merged_tokens:', node.GetMergedTokens(), '}')
			for weight, node_neighbor in node.GetWeightedNeighbors():
				print('	   weight: ', weight, ' neighbor: ', node_neighbor.GetToken())

	def PrintNodeCount(self):
		print('Graph: ')
		print('Number of nodes: ', len(self.token_to_node))

	def GetNodesWithSmallFreq(self, freq):
		# return token list with freq less than given freq 
		tokens_with_small_freq = list()
		for token, node in self.token_to_node.items():
			node_freq = node.GetTokenFreq()
			if node_freq < freq:
				tokens_with_small_freq.append(token)
		return tokens_with_small_freq
	
	def Merge_TokenA_Into_TokenB(self, token_a, token_b):
		node_a = self.GetNode(token_a)
		node_b = self.GetNode(token_b)

		# Already merged together
		if node_a == node_b:
			return
		
		self.num_merges += 1
		
		# Save a copy here so that we are not iterating over things while we are deleting.
		node_a_neighbor_tokens_to_weight = dict()
		for weight, node_a_neighbor_token in node_a.GetWeightedNeighborTokens():
			node_a_neighbor_tokens_to_weight[node_a_neighbor_token] = weight

		# Remove any edges to NodeA
		for node_a_neighbor_token, weight in node_a_neighbor_tokens_to_weight.items():
			self.RemoveEdge(node_a_neighbor_token, token_a)

		# Add edges form NodeA neighbors to NodeB
		# (accumulate the weight if the edge already exists)
		for node_a_neighbor_token, weight in node_a_neighbor_tokens_to_weight.items():
			# make sure we do not add an edge from token_b to token_b
			if node_a_neighbor_token != token_b:
				self.AddEdgeOrAccumulateEdgeWeight(node_a_neighbor_token, token_b, weight)

		# Keep track of merged nodes to help debugging
		node_b.AddMergedToken(token_a)
		for node_a_merged_token in node_a.GetMergedTokens():
			node_b.AddMergedToken(node_a_merged_token)

		# Remove NodeA
		logging.info('merging ' + token_a + ' into ' + token_b + ', removing ' + token_a)
		self.token_to_node.pop(token_a)
		self.token_to_merged_token[token_a] = token_b
		

	def GetTokensToTotalEdgeWeightAccending(self):
		token_to_total_edge_weight = dict()
		for token, node in self.token_to_node.items():
			token_to_total_edge_weight[token] = node.GetTotalEdgeWeight()
		return sorted(token_to_total_edge_weight.items(), key=lambda x: x[1]) #ASC1[(token, degree), ..]	
	
	def SaveGraphToJson(self, output_filename):
		json_edges = []
		json_nodes = []
		for node_token, node in self.token_to_node.items():
			json_node = {
				"token": node.GetToken(),
				"merged_tokens": node.GetMergedTokens(),
			}
			json_nodes.append(json_node)

			for weight, node_neighbor_token in node.GetWeightedNeighborTokens():
				json_edge = {
					"token_a": node_token,
					"token_b": node_neighbor_token,
					"weight": weight
				}
				json_edges.append(json_edge)

		json_graph = {
			"nodes": json_nodes,
			"edges": json_edges,
		}

		with open(output_filename, 'w', encoding='utf-8') as output_file:
			json.dump(json_graph, output_file, ensure_ascii=False, indent=2)

	def LoadGraphFromJson(self, input_filename):
		json_graph = None
		with open(input_filename, 'r', encoding='utf-8') as input_file:
			json_graph = json.load(input_file)

		# adding nodes back
		for node in json_graph["nodes"]:
			node_token = node["token"]
			node_merged_tokens = node["merged_tokens"]

			self.AddNode(node_token)
			node = self.GetNode(node_token)
			for node_merged_token in node_merged_tokens:
				node.AddMergedToken(node_merged_token)
				self.token_to_merged_token[node_merged_token] = node_token

		# adding edges back
		for edge in json_graph["edges"]:
			token_a = edge["token_a"]
			token_b = edge["token_b"]
			weight = edge["weight"]
			self.AddEdge(token_a, token_b, weight)
