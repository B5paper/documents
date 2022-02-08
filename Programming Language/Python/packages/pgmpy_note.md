# Pgmpy Note

一些术语：

* PGM: Probabilistic Graphical Models
* CPD: Conditional Probabilisy Distributions
* DAG: Directed acyclic graph

## Bayesian Network

* `pgmpy.models.BayesianModel(edges)`

    Parameters:

    * `edges`: list of tuples

        Example: `[('D', 'G'), ('I', 'G')]`

* `pgmpy.factors.discrete.TabularCPD(variable=None, variable_card=None, values=None, evidence=None, evidence_card=None, state_names=None)`

    Parameters:

    * `variable`: `str`
    * `variable_card`: `int`, 指随机变量有多少个离散值
    * `values`: list of list, 每行代表一个随机变量的取值，每列代表一个证据变量的组合。
    * `evidence`: list of str，条件变量的名称
    * `evidence_card`：list of int，条件变量的度
    * `state_names`（optional）：`dict`, `{str: list}`，其中 list 是随机变量的可取值

* `model.get_cpds(name=None)`

    * `name` (optional): 如果不填的话，返回所有的 cpd，如果填变量的名字的话，返回的是指定变量的 cpd。

* `model.check_model()`

* `model.add_cpds(cpd_1, cpd_2, ...)`

* `model.get_cardinality(var_name)`

    获得指定变量的度。

* `model.local_independencies(var_names)`

    Parameters:

    * `var_names`: 可以是 str，也可以是 list of str。


```python
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

cpd_d
```

* `pgmpy.inference.VariableElimination(model)`

    Example:

    ```python
    from pgmpy.inference import VariableElimination
    infer = VariableElimination(model)
    g_dist = infer.query(['G'])
    print(g_dist)

    print(infer.query(['G'], evidence={'D': 'Easy', 'I': 'Intelligent'}))

    # 寻找 map 假设
    infer.map_query(['G'])
    infer.map_query(['G'], evidence={'D': 'Easy', 'I': 'Intelligent'})
    infer.map_query(['G'], evidence={'D': 'Easy', 'I': 'Intelligent', 'L': 'Good', 'S': 'Good'})
    ```

## API

### DAG (Directed Acyclic Graph)

#### Class `DAG`

* `class pgmpy.base.DAG(ebunch=None, latents={})`

    Base class for all Directed Graphical Models.

    Each node in the graph can represent either a random variable, Factor, or a cluster of random variables. Edges in the graph represent the dependencies between these.

    Parameters:

    * `data`: Data to initialize graph. If `data=None` an empty graph is created. The data can be an edge list or any Networkx graph object.

    ```python
    from pgmpy.base import DAG
    G = DAG()

    # add nodes
    G.add_node(node='a')
    G.add_nodes_from(nodes=['a', 'b'])

    # add edges
    G.add_edge(u='a', v='b')
    G.add_edges_from(ebunch=[('a', 'b'), ('b', 'c')])

    G.nodes()
    G.edges()
    ```

    If some edges connect nodes not yet in the model, the nodes are added automatically.

    shortcuts:

    ```python
    'a' in G  # check if node in graph
    len(G)  # number of nodes in graph
    ```

* `active_trail_nodes(variables, observed=None, include_latents=False)`

    Returns a dictionary with the given variables as keys and all the nodes reachable from that respective variable as values.

    Parameters: 

    * `variables`: str or array like

    * `observed`: list of nodes (optional)

    * `include_latents` (boolean): Whether to include the latent variables in the returned active trail nodes.

    ```python
    model.active_trail_nodes('D', observed='G')
    ```

* `add_edge(u, v, weight=None)`

    The nodes `u` and `v` will be automatically added if they are not already in the graph.

* `add_edges_from(ebunch, weights=None)`

    Parameters:

    * `ebunch` (container of edges): Each edge given in the container will be added to the graph. The edges must be given as 2-tuples `(u, v)`.

    * `weights` (list, tuple): A container of weights (int, float). The weight value at index i is associated with the edge at index i.

* `add_node(node, weight=None, latent=False)`

    Parameters:

    * `latent` (bollean): Specifies whether the variable is latent or not.

* `add_nodes_from(nodes, weights=None, latent=False)`

    Parameters:

    * `latent` (list, tuple): A container of boolean. The value at index i tells whether the node at index i is latent or not.

* `do(nodes, inplace=False)`

    Applies the do operator to the graph and return a new DAG with the transformed graph.

    The do-operator, `do(X=x)` has the effect of removing all edges from the parents of `X` and setting `X` to the given value `x`.

    Parameters:

    * `nodes` (list, array-like): The names of the nodes to apply the do-operator for.

    * `inplace` (boolean): If `True`, makes the changes to the current object, otherwise returns a new instance.

    Examples:

    ```python
    graph = DAG()
    graph.add_edges_from([('X', 'A'), ('A', 'Y'), ('A', 'B')])
    graph_do_A = graph.do('A')
    # Which we can verify is missing the edges we would expect.
    graph_do_A.edges()
    ```

* `get_ancestral_graph(nodes)`

    Returns the ancestral graph of the given nodes. The ancestral graph only contains the nodes which are ancestors of atleast one of the variables in node.

    Examples:

    ```python
    dag = DAG([('A', 'C'), ('B', 'C'), ('D', 'A'), ('D', 'B')])
    anc_dag = dag.get_ancestral_graph(nodes=['A', 'B'])
    anc_dag.edges()
    ```

* `get_children(node)`

    Returns a list of children of node.

* `get_immoralities()`

    Finds all the immoralities in the model.

    A v-structure `X -> Z <- Y` is an immorality if there is no direct edge between X and Y. （在这个例子中，将返回`{('X', 'Y')}`）

* `get_independencies(latex=False, include_latents=False)`

    Computes independencies in the DAG, by checking d-seperation.

* `get_leaves()`

* `get_markov_blanket(node)`

    Returns a markov blanket for a random variable. In the case of Bayesian Networks, the markov blanket is the set of node's parents, its children and its children's other parents.

* `get_parents(node)`

* `static get_random(n_nodes=5, edge_prob=0.5, latents=False)`

    Returns a randomly generated DAG with `n_nodes` number of nodes with edge probability being `edge_prob`.

    Parameters:

    * `latents`: If True, includes latent variables in the generated DAG.

* `get_roots()`

* `is_dconnected(start, end, observed=None)`

    Returns True if there is an active trail (i.e. d-connection) between `start` and `end` node given that `observed` is observed.

    Parameters:

    * `observed` (list, array-like, optional): If given the active trail would be computed assuming these nodes to be observed.

* `is_iequivalent(model)`

    Checks whether the given model is I-equivalent.

    Two graphs G1 and G2 are said to be I-equivalent if they have same skeleton and have same set of immoralities.

    Examples:

    ```python
    G = DAG()
    G.add_edges_from([('V', 'W'), ('W', 'X'), ('X', 'Y'), ('Z', 'Y')])
    G1 = DAG()
    G1.add_edges_from([('W', 'V'), ('X', 'W'), ('X', 'Y'), ('Z', 'Y')])
    G.is_iequivalent(G1)  # True
    ```

* `local_independencies(variables)`

    Return an instance of independencies containing the local independencies of each of the variables.

    Examples:

    ```python
    student = DAG()
    student.add_edges_from([('diff', 'grade'), ('intel', 'grade'), ('grade', 'letter'), ('intel', 'SAT')])
    ind = student.local_independencies('grade')
    ind  # (grade _|_ SAT | diff, intel)
    ```

* `minimal_dseparator(start, end)`

    Finds the minimal d-separating set for start and end.

    Examples:

    ```python
    dag = DAG([('A', 'B'), ('B', 'C')])
    dag.minimal_dseparator(start='A', end='C')  # {'B'}
    ```

* `moralize()`

    Removes all the immoralities in the DAG and creates a moral graph (UndirectedGraph).

    A v-structure X -> Z <- Y is an immorality if there is no directed edge between X and Y.

    Examples:

    ```python
    G = DAG(ebunch=[('diff', 'grade'), ('intel', 'grade')])
    moral_graph = G.moralize()
    moral_graph.edges()  # EdgeView([('intel', 'grade'), ('intel', 'diff'), ('grade', 'diff')])
    ```

* `to_daft(node_pos='circular', latex=True, pgm_params={}, edge_params={}, node_params={})`

    Return a daft object which can be rendered for publication quality plots. The returned object's render method can be called to see the plots.

    Parameters:

    * `node_pos` (str or dict): `{'circular' | 'kamada_kawai' | 'planar' | 'random' | 'shell' | 'sprint' | 'spectral' | 'spiral'}`

        If dict, should be of the form `{node: (x coordinate, y coordinate)}`.

    * `latex` (boolean): Whether to use latex for rendering the node names.

    * `pgm_params` (dict, optional): These params are passed to `daft.PGM` initializer. Should be of the form `{param_name: param_value}`

    * `edge_params` (dict, optional): These params are passed to `daft.add_edge` method. Should be of the form: `{(u1, v1): {param_name: param_value}, (u2, v2): {...}}`

    * `node_params` (dict, optional): Any additional node parameters that need to be passed to `daft.add_node` method. Should be of the form: `{node1: {param_name: param_value}, node2: {...}}`

* `to_pdag()`

    Returns the PDAG (Partially oriented DAG) of the DAG.

#### Class PDAG

* `class pgmpy.base.PDAG(directed_ebunch=[], undirected_ebunch=[], latents=[])`

    在这个类中，undirected edges 使用两个有向箭头表示。比如 X - Y 会被表示为 X -> Y 和 X <- Y。

* `copy()`

    Returns a copy of the object instance.

* `to_dag(required_edges=[])`

    Parameters:

    * `required_edges(list, array-list of 2-tuples)`: The list of edges that should be included in the DAG.

### Bayesian Network

#### Class BayesianNetwork

* `class pgmpy.models.BayesianNetwork.BayesianNetwork(ebunch=None, latents={})`

* `add_cpds(*cpds)`

* `add_edgs(u, v, **kwargs)`

* `check_model()`

* `copy()`

    Returns a copy of the model.

* `do(nodes, inplace=False)`

    Applies the do operation. The do operation removes all incoming edges to variables in nodes and marginalizes their CPDs to only contain the variable itself.

* `fit(data, estimator=None, state_names=[], complete_samples_only=True, n_jobs=-1, **kwargs)`

    Estimate the CPD for each variable based on a given data set.

    Parameters:

    * `data` (pandas `DataFrame` object): `DataFrame` object with column names identical to the variable names of the network. (If some values in the data are missing the data cells should be set to `numpy.NaN`. Note that pandas converts each column containing `numpy.NaN`'s to dtype `float`.)

* `fit_update(data, n_prev_samples=None, n_jobs=-1)`

* `get_cardinality(node=None)`

    Returns the cardinality of the node. If node is not specified returns a dictionary with the given variable as keys and their respective cardinality as values.

* `get_cpds(node=None)`

* `get_factorized_product(latex=False)`

* `get_markov_blanket(node)`

* `static get_random(n_nodes=5, edge_prob=0.5, n_states=None, latents=False)`

* `is_imap(JPD)`

    Checks whether the bayesian model is Imap of given JointProbabilityDistribution.

* `static load(filename, filetype='bif')`

* `save(filename, filetype='bif')`

    Parameters:

    * `filetype`: `{'bif' | 'uai' | 'xmlbif'}`

* `predict(data, stochastic=False, n_jobs=-1)`

    Predicts states of all the missing variables.

    Parameters:

    * `data` (pandas `DataFrame` object): A `DataFrame` object with column names same as the variables in the model.

    * `stochastic`: If True, does prediction by sampling from the distribution of predicted variable(s). If False, returns the states with the highest probability value (i.e MAP) for the predicted variable(s).

    * `n_jobs`: The number of CPU cores to use. If `-1`, uses all available cores.

* `predict_probability(data)`

    Predicts probabilities of all states of the missing variables.

* `remove_cpds(*cpds)`

    Removes the cpds that are provided in the argument.

* `remove_node(node)`

    Remove node from the model.

    Removing a node also removes all the associated edges, removes the CPD of the node and marginalizes the CPDs of it's children.

* `remove_nodes_from(nodes)`

* `simulate(n_samples=10, do=None, evidence=None, virtual_evidence=None, virtual_intervention=None, include_latents=False, partial_samples=None, seed=None, show_progress=True)`

    Simulates data from the given model. Internally uses methods from `pgmpy.sampling.BayesianModelSampling` to generate the data.

    Return a dictionary mapping each node to its list of possible states.

    * `n_samples`: The number of data samples to simulate from the model.

    * `do` (dict): The interventions to apply to the model. `dict` should be of the form `{variable_name: state}`.

    * `evidence` (dict): Observed evidence to apply to the model. `dict` should be of the form `{variable_name: state}`.

    * `virtual_evidence` (list): Probabilistically apply evidence to the model. `virtual_evidence` should be a list of `pgmpy.factors.discrete.TabularCPD` objects specifying the virtual probabilities.

    * `virtual_intervention` (list): Also known as soft intervention. `virtual_intervention` should be a list of `pgmpy.factors.discrete.TabularCPD` objects specifying the virtual/soft intervention probabilities.

    * `include_latents`: Whether to include the latent variable values in the generated samples.

    * `partial_samples` (`pandas.DataFrame`): A pandas dataframe specifying samples on some of the variables in the model. If specified, the sampling procedure uses these sample values, instead of generating them. `partial_samples.shape[0]` must be equal to `n_samples`.

    * `seed`: the seed for `numpy.random`

    * `show_progress`: whether to show the progress bar

    Examples:

    ```python
    from pgmpy.utils import get_example_model
    
    # Simulation without any evidence or intervention
    model = get_example_model('alarm')
    model.simulate(n_samples=10)

    # simulation with the hard evidence: MINVOLSET = HIGH
    model.simulate(n_samples=10, evidence={"MINVOLSET": "HIGH"})

    # simulation with hard intervention: CVP = LOW
    model = simulate(n_samples=10, do={'CVP': 'LOW'})

    # simulation with virtual/soft evidence: p(MINVOLSET=LOW) = 0.8, p(MINVOLSET=HIGH) = 0.2, p(MINVOLSET=NORMAL) = 0:
    virt_evidence = [TabularCPD('MINVOLSET', 3, [[0.8], [0.0], [0.2]], state_names={'MINVOLSET': ['LOW', 'NORMAL', 'HIGH']})]
    model.simulate(n_samples, virtual_evidence=virt_evidence)

    # Simulation with virtual/soft intervention: p(CVP=LOW) = 0.2, p(CVP=NORMAL) = 0.5, p(CVP=HIGH) = 0.3
    virt_intervention = [Tabular('CVP', 3, [[0.2], [0.5], [0.3]], state_names={'CVP': ['LOW', 'NORMAL', 'HIGH']})]
    model.simulate(n_samples, virtual_intervention=virt_intervention)
    ```

    * `to_junction_tree()`

        Creates a junction tree (or clique tree) for a given bayesian model.

    * `to_markov_model()`

        Converts bayesian model to markov model.