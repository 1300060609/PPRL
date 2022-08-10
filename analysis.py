'''
consensus cluster for PPRL embedded vectors
pca representation
pathway finding
protein finding and visualization
'''
from out_put import *
from train import *
from sklearn import decomposition as skldec
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# set configures
dirs = ['2020-11-28_17:24:14']

k = 6
for model_dir in dirs:
    print(model_dir)
    config = json.load(open(os.path.join(model_dir, 'configuration.json'), 'r'))
    input_dir = config['input_dir']

    embeddings = json.load(open(os.path.join(model_dir, 'entities_to_embeddings.json')))
    df = pd.DataFrame(embeddings)
    print(input_dir)
    samples = pd.read_csv(os.path.join(input_dir, 'samples.csv'), usecols=[0], header=None)

    m = df[samples[0]]
    m.to_csv(os.path.join(model_dir, 'embedding_matrix.csv'))
    prior = config['prior_weight']
    train = TrainPPRL(config)
    model = train.model
    path_to_model = os.path.join(model_dir, 'trained_model.pkl')
    state_dict = torch.load(path_to_model)
    model.load_state_dict(state_dict)
    results = prepare_output(train, model)
    # prepare to evaluate

    sample_names = pd.DataFrame(train.Eca)[0].drop_duplicates().values
    sample_indices = [train.vocab[i] for i in sample_names]
    truth_cluster = pd.read_csv(os.path.join(input_dir, 'samples.csv'), header=None, index_col=0)
    colors = [truth_cluster.loc[i][1] for i in sample_names]
    labels = list(truth_cluster[2].unique())
    dt = model.entity_embeddings.weight.data
    df = pd.DataFrame(dt, dtype='float').iloc[sample_indices]

    # consensus survival
    os.chdir(os.path.join(os.getcwd(), model_dir))
    os.system('Rscript /Users/sdz/PycharmProjects/PPRL7/consensus_survival_our.R embedding_matrix.csv\
     /Users/sdz/PycharmProjects/PPRL7/input_cptac_6_na_exp/survival_hcc.csv 15 hc pearson %d' % k)
    os.chdir('/Users/sdz/PycharmProjects/PPRL7')

    # # Plot PCA according to the two largest principal components
    pca = skldec.PCA(n_components=0.95)  # Select the proportion of 95% variance
    pca.fit(df)  # In principal component analysis, each line is an input data
    result = pca.transform(df)
    X = result[:, 0]
    Y = result[:, 1]
    Z = result[:, 2]
    scatter_figure_3d(model_dir, 'result/fig_pca.pdf', pca, X, Y, Z, colors, labels)
    #
    #
    #
    # # #
    # # PCA visualization of clustering results after clustering
    consensus_results = pd.read_csv(os.path.join(model_dir, 'result/ConsensusResult.csv'), index_col=0)
    new_sample_names = [i.replace(':', '.').replace('-', '.').replace(' ', '.') for i in sample_names]
    new_colors = list(consensus_results.loc[new_sample_names]['k=%d' % k].values)
    new_labels = ['Prior_based_cluster%d' % (i + 1) for i in range(k)]
    scatter_figure_3d(model_dir, 'result/fig_pca_cluster.pdf', pca, X, Y, Z, new_colors, new_labels)

    # # #
    # # Prepare pathways and protein discovery
    clusters = pd.read_csv(os.path.join(model_dir, 'result/ConsensusResult.csv'), sep=',')
    with open('pathway_proteins.json', 'r') as f:
        pathway_proteins = json.load(f)
    entity_embeddings = pd.DataFrame(results[ENTITY_TO_EMBEDDING]).T
    pca = skldec.PCA(n_components=3)  # Select the proportion of 95% variance
    pca.fit(entity_embeddings)  # In principal component analysis, each line is an input data
    result = pca.transform(entity_embeddings)
    index = entity_embeddings.index
    result_df = pd.DataFrame(result, index=index)

    # display pathway
    pathway_name = 'p53 signaling pathway'
    proteins = pathway_proteins[pathway_name]
    proteins2scatter = {}
    for protein in proteins:
        if protein in result_df.index:
            proteins2scatter[protein] = result_df.loc[protein]
    proteins2scatter_df = pd.DataFrame(proteins2scatter).T
    scatter_figure_3d_protein(model_dir, '%s.pdf' % pathway_name, pca, proteins2scatter_df)
    #
    # # # WB-ratio
    # array = m.T.values
    # colors4wb = []
    # for i in m.columns:
    #     colors4wb.append(clusters['k=%d' % k][np.where(clusters['sample']==i)[0]].values[0])
    # wb_ratio = WB_ratio(array, colors4wb)
    # with open(os.path.join(model_dir, 'result/wb_ratio.txt'), 'w+') as f1:
    #     json.dump(wb_ratio, f1)

    # # Pathway discovery
    type_pathways = find_significant_pathway_for_each_sample_type(model_dir,
                                                                  entity_embeddings=results[ENTITY_TO_EMBEDDING], k=k)
    type_pathways.to_csv(os.path.join(model_dir, 'result/pathway_results.csv'))
    type_pathway = []
    data4scatter = {}
    result_emb = {}
    labels4wb = []
    for i in range(k):
        pathways = type_pathways.loc[np.where(type_pathways['cluster'] == str(i + 1))].sort_values('correlation',
                                                                                                   ascending=False).iloc[
                   :3, :]['pathway'].values
        samples = clusters['sample'][np.where(clusters['k=%d' % k] == i + 1)[0]]
        sample_pca = result_df.loc[samples].mean()
        sample_emb = entity_embeddings.loc[samples].mean()
        data4scatter['Prior_based_cluster' + str(i + 1)] = sample_pca
        result_emb['Prior_based_cluster' + str(i + 1)] = sample_emb
        labels4wb.append(i)
        proteins = []

        for pathway in pathways:
            proteins = pathway_proteins[pathway]
            proteins = list(set(proteins).intersection(set(index)))
            proteins_pca = result_df.loc[proteins].mean()
            proteins_emb = entity_embeddings.loc[proteins].mean()
            data4scatter[pathway] = proteins_pca
            if pathway not in result_emb:
                labels4wb.append(i)
                result_emb[pathway] = proteins_emb

    data4scatter_df = pd.DataFrame(data4scatter).T
    result_emb_df = pd.DataFrame(result_emb).T
    wb_ratio_path = WB_ratio(np.array(result_emb_df.values), labels4wb)
    with open(os.path.join(model_dir, 'result/wb_ratio_pathways.txt'), 'w+') as f1:
        json.dump(wb_ratio_path, f1)

    # #
    result_emb_df.to_csv(os.path.join(model_dir, 'result/pathway_embeddings.csv'))
    data4scatter_df.to_csv(os.path.join(model_dir, 'result/pathway_pca.csv'))
    scatter_figure_3d_pathway(model_dir, 'result/fig_pca_pathways.pdf', pca, data4scatter_df)
    #
    # # #protein discovery
    type_proteins = find_significant_proteins_for_each_sample_type(model_dir,
                                                                   entity_embeddings=results[ENTITY_TO_EMBEDDING], k=k)
    type_proteins.to_csv(os.path.join(model_dir, 'result/protein_results.csv'))
    type_protein = []
    data2scatter = {}
    result_emb_protein = {}
    for i in range(k):
        proteins = type_proteins.loc[np.where(type_proteins['cluster'] == str(i + 1))].sort_values('correlation',
                                                                                                   ascending=False).iloc[
                   :3, :]['protein'].values
        samples = clusters['sample'][np.where(clusters['k=%d' % k] == i + 1)[0]]
        sample_pca = result_df.loc[samples].mean()
        sample_emb = entity_embeddings.loc[samples].mean()
        data2scatter['Prior_based_cluster' + str(i + 1)] = sample_pca
        result_emb_protein['Prior_based_cluster' + str(i + 1)] = sample_emb
        # sample_pca = result_df.loc[samples]
        # for x in sample_pca.index:
        #     data2scatter[x]=sample_pca.loc[x]
        for protein in proteins:
            protein_pca = result_df.loc[protein]
            protein_emb = entity_embeddings.loc[protein]
            data2scatter[protein] = protein_pca
            result_emb_protein[protein] = protein_emb
    data2scatter_df = pd.DataFrame(data2scatter).T
    result_emb_protein_df = pd.DataFrame(result_emb_protein).T
    data2scatter_df.to_csv(os.path.join(model_dir, 'result/protein_pca.csv'))
    result_emb_protein_df.to_csv(os.path.join(model_dir, 'result/protein_embeddings.csv'))
    scatter_figure_3d_protein(model_dir, 'result/fig_pca_protein.pdf', pca, data2scatter_df)

    # if input_dir=='input_cptac':
    # hotmap protein
    f1 = pd.read_csv(os.path.join(input_dir, 'profiles.csv'), sep=',', index_col=0)

    f1 = f1[~f1.index.duplicated(keep='first')]
    f1 = f1  # .iloc[:,-101:]
    f1.index = f1.index.str.upper()
    d1 = f1.fillna(0)
    consensus_results = pd.read_csv(os.path.join(model_dir, 'result/ConsensusResult.csv'), sep=',')
    # print(df)
    sample_types = {}
    embeddings = {}
    cluster_ex = pd.DataFrame()
    for i in consensus_results.index:
        line = consensus_results.iloc[i]
        sample = line['sample']
        type = line['k=%d' % k].astype('str')
        sample_types.setdefault(type, []).append(sample)

    samples2hotmap = []
    proteins2hotmap = []
    for i in range(k):
        proteins = type_proteins.loc[np.where(type_proteins['cluster'] == str(i + 1))].sort_values('correlation',
                                                                                                   ascending=False).iloc[
                   :3, :]['protein'].values
        for protein in proteins:
            if protein in d1.index:
                proteins2hotmap.append(protein)
    cols = []
    for type in sample_types:
        type_ex = f1[sample_types[type]].mean(axis=1)
        cluster_ex = pd.concat([cluster_ex, type_ex], axis=1)
        cols.append(eval(type))
    cluster_ex.columns = cols
    cluster_ex.sort_index(axis=1, inplace=True)
    # plt.figure(figsize=(10,8))
    f, ax = plt.subplots(figsize=(9, 6))
    proteins_heat_data = cluster_ex.loc[proteins2hotmap].T
    proteins_heat_data.to_csv(os.path.join(model_dir, 'result/hotmap_protein.csv'))
    ax = sns.heatmap(proteins_heat_data.fillna(0), xticklabels=1, cbar=True, cmap='RdBu_r')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Mean SD')
    ax.set_xlabel('Proteins')
    ax.set_ylabel('Prior-based clusters')
    fig = ax.get_figure()
    fig.savefig(os.path.join(model_dir, 'result/fig_hotmap_protein.pdf'), bbox_inches='tight')
