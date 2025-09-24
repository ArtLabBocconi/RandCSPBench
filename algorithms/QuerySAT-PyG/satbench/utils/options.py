def add_model_options(parser):
    parser.add_argument('--graph', type=str, choices=['lcg', 'vcg'], default='lcg', help='Graph construction')
    parser.add_argument('--init_emb', type=str, choices=['learned', 'random', 'ones'], default='ones', help='Embedding initialization')
    parser.add_argument('--model', type=str, choices=['querysat'], default='querysat', help='GNN model')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of embeddings and hidden states')
    parser.add_argument('--n_iterations', type=int, default=32, help='Number of iterations for message passing')    
    parser.add_argument('--activation', type=str, default='leaky_relu', help='Activation function in all MLPs')
