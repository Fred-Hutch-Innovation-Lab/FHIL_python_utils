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

def grid_plot_adata(
    adata_dict,
    plot_func,
    color_keys,
    plot_kwargs=None,
    per_row=True,
    figsize=(4, 4),
    pass_show=True 
):
    """
    Wrap plots from multiple sc objects into a grid.
    """

    if plot_kwargs is None:
        plot_kwargs = {}

    n_rows = len(adata_dict) if per_row else len(color_keys)
    n_cols = len(color_keys) if per_row else len(adata_dict)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_i, (obj_name, adata) in enumerate(adata_dict.items() if per_row else color_keys):
        for col_i, color in enumerate(color_keys if per_row else adata_dict.items()):
            i, j = (row_i, col_i) if per_row else (col_i, row_i)

            current_adata = adata if per_row else adata_dict[color]
            current_color = color if per_row else obj_name
            ax = axes[i, j]

            call_kwargs = {
                "color": current_color,
                "ax": ax,
                **plot_kwargs
            }

            if pass_show:
                call_kwargs["show"] = False

            plot_func(current_adata, **call_kwargs)
            ax.set_title(f"{obj_name} - {current_color}" if per_row else f"{color} - {obj_name}")

    plt.tight_layout()
    plt.show()
