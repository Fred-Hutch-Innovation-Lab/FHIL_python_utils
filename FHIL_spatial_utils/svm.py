def svm_feature_weighting_heatmap(model, n=5, figsize=(8, 12)):
    '''
    Get the top informative features per class for an SVM
    '''
    feature_names = model['svc'].feature_names_in_
    class_labels = model['svc'].classes_
    n_classes = len(class_labels)
    class_pairs = list(combinations(range(n_classes), 2))  # order matches .coef_
    
    # Accumulate weights for each class
    coefs = model['svc'].coef_
    n_features = coefs.shape[1]
    class_feature_weights = {cls: np.zeros(n_features) for cls in range(n_classes)}
    counts = {cls: 0 for cls in range(n_classes)}
    
    # Accumulate absolute weights from pairwise classifiers
    for coef, (i, j) in zip(coefs, class_pairs):
        class_feature_weights[i] += np.abs(coef)
        class_feature_weights[j] += np.abs(coef)
        counts[i] += 1
        counts[j] += 1
    top_features = {}
    for cls in range(n_classes):
        # avg_weights = class_feature_weights[cls] / counts[cls]
        weights = class_feature_weights[cls].A1
        top_idx = np.argsort(weights)[::-1][:n]
        top_features[class_labels[cls]] = {
            feature_names[i]: weights[i] for i in top_idx
        }
    
    # Convert to DataFrame
    plotdata = pd.DataFrame(top_features).fillna(0)
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(plotdata, cmap='viridis', fmt=".2f", cbar_kws={'label': 'Avg |Weight|'})
    plt.title(f'Top {n} features per class from OvO SVM')
    ax.set_xticks([x + 0.5 for x in range(plotdata.shape[1])], plotdata.columns.values, rotation=45, ha='right', rotation_mode='anchor')
    plt.ylabel('Class')
    plt.xlabel('Feature')
    plt.tight_layout()
    plt.show()

def svm_classes_from_proba(probs,
                           classes,
                           threshold=0.5, # Minimum required confidence
                           margin=0.1     # Allow second-best class if within this range of the top
                          ):
    '''
    Get the top predicted class(es) from a SVM predict_proba call
    '''
    pred_classes = []
    pred_classes_simplified = []
    for prob in probs:
        top_idx = np.argmax(prob)
        top_prob = probs[top_idx]
    
        # Mask out top index to find second-best
        second_idx = np.argsort(prob)[-2]
        second_prob = probs[second_idx]
    
        if top_prob < threshold:
            pred_classes.append('unknown')
            pred_classes_simplified.append('unknown')
        elif (top_prob - second_prob) <= margin:
            pred_classes.append(f"{classes[top_idx]} | {classes[second_idx]}")
            pred_classes_simplified.append('mixed')
        else:
            pred_classes.append(classes[top_idx])
            pred_classes_simplified.append(classes[top_idx])
    pd.Series(pred_classes_simplified).value_counts()
    return pred_classes, pred_classes_simplified

def annotate_clusters_by_consensus(
    obj,
    cluster_column = 'over_clustering',
    annotation_column = 'svm_predicted_class',
    proportion_threshold = 0.5,
    margin = 0.15,
    output_column = 'overclustering_consensus_annotation'
):
    '''
    Aggregate annotation predictions by cluster. 
    A cluster must have greater than `proportion_threshold` ratio of cells
    with a single annotation to be labelled. A cluster will be annotated with
    two labels if there is a second label within `margin` of the 1st most
    abundant label. 
    '''
    data = (
        obj.obs.groupby(cluster_column)[annotation_column]
        .value_counts(normalize=True)
        .groupby(level=0)
        .head(2)
        .reset_index(name='count')
    )
    assigned_labels = {}
    for group, group_df in data.groupby(cluster_column):
        top_values = group_df.sort_values('count', ascending=False).reset_index(drop=True)
        top1 = top_values.loc[0]
    
        if top1['count'] >= proportion_threshold:
            assigned_labels[group] = top1[annotation_column]
    
        elif len(top_values) > 1:
            top2 = top_values.loc[1]
    
            # Check margin condition
            if abs(top1['count'] - top2['count']) <= margin and 'unknown' not in {top1[annotation_column], top2[annotation_column]}:
                values = sorted([top1[annotation_column], top2[annotation_column]])
                assigned_labels[group] = f"{values[0]} | {values[1]}"
            else:
                assigned_labels[group] = 'unknown'
    
        else:
            assigned_labels[group] = 'unknown'
    obj.obs[output_column] = obj.obs[cluster_column].map(assigned_labels)