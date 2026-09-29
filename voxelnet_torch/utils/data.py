def infinite_loader(loader, start_epoch: int = 0):
    """
    Get an infinite stream of batches from a data loader.

    Useful for iteration-based training where you want to iterate
    for a fixed number of iterations rather than epochs.

    The DataLoader will automatically reshuffle (if shuffle=True)
    at the start of each epoch.

    Args:
        loader: PyTorch DataLoader
        start_epoch: Epoch number to start from (e.g. when resuming), used for
            DistributedSampler.set_epoch

    Yields:
        Batches from the loader, infinitely cycling through epochs

    Example:
        >>> train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        >>> train_iter = iter(infinite_loader(train_loader))
        >>> for i in range(10000):  # Train for 10000 iterations
        >>>     batch = next(train_iter)

    With a DistributedSampler, `set_epoch` is called at the start of every
    epoch so that each epoch uses a different shuffle on all processes.
    """
    epoch = start_epoch
    sampler = getattr(loader, 'sampler', None)
    while True:
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1
