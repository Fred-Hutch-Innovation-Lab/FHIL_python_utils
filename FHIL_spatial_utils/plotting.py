def plot_confusion_matrix(x, y, normalize=False, figsize=(8, 6), cmap='Blues', annot=True):
    """
    Generic confusion matrix for any X and Y, allowing different classes.

    Parameters:
        x (array-like):
        y (array-like): 
        normalize (bool): If True, normalize rows (true classes) to sum to 1.
        figsize (tuple): Size of the figure.
        cmap (str): Colormap for heatmap.
    """
    x = pd.Categorical(x)
    y = pd.Categorical(y)

    # Union of all labels
    # all_labels = sorted(set(y_true) | set(y_pred))
    
    # Create a confusion matrix DataFrame initialized to zero
    cm = pd.DataFrame(
        0, index=y.categories, columns=x.categories, dtype=float if normalize else int
    )

    # Populate the confusion matrix
    for true, pred in zip(x, y):
        cm.loc[pred, true] += 1

    if normalize:
        cm = cm.div(cm.sum(axis=1), axis=0).fillna(0) * 100

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm,
                annot=annot,
                fmt=".0f" if normalize else "d",
                cmap=cmap,
                cbar=True,
                cbar_kws={'label': '% of Label 2'} if normalize else None
    )
    ax.set_xticks([x + 0.5 for x in range(len(x.categories))], x.categories, rotation=45, ha='right', rotation_mode='anchor')
    plt.xlabel("Label 1")
    plt.ylabel("Label 2")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.show()
