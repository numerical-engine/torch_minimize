import torch


def line_armijo(
        func:"Function",
        x:torch.Tensor,
        tau:float,
        alpha:float,
        f_current:torch.float,
        grad_f_current:torch.Tensor,
        sigma:float = 0.5,
        alpha_min:float = 0.,
        d:torch.Tensor = None,
        target_indices:tuple[int] = None,)->float:
    """バックトラッキング法によるArmijo条件を満たす実数alphaを出力する。

    dを降下方向としたととき、入力xはx + alpha*dに更新される。

    Args:
        func (Function): 目的関数。出力はスカラー値である必要がある。
        x (torch.Tensor): 現時点の候補解。shapeは(変数の次元数,)である必要がある。
        f_current (torch.float): 現時点の候補解xにおける目的関数の値。
        grad_f_current (torch.Tensor): 現時点の候補解xにおける目的関数の勾配。shapeは(len(target_indices),)である必要がある。
        target_indices (tuple[int]): 入力のうち更新対象のインデックスのタプル。例えば、xが3次元でtarget_indices=(0, 2)とすると、x[0]とx[2]のみ最適化対象。
        sigma (float): Armijo条件のパラメータ。0 < sigma < 1である必要がある。
        tau (float): バックトラッキング法のパラメータ。0 < tau < 1である必要がある。
        alpha (float): 初期ステップサイズ。alpha_min < alphaである必要がある。
        alpha_min (float, optional): ステップサイズの下限。
        d (torch.Tensor, optional): 降下方向。shapeは(len(target_indices),)である必要がある。Noneの場合はfuncの最急降下方向とする。
    Returns:
        float: Armijo条件を満たすステップサイズalpha。alpha_minより大きいalphaで条件を満たすものが見つからなかった場合はalpha_minを返す。
    Raises:
        ValueError: sigmaが0以上1以下の値でない場合。
        ValueError: tauが0以上1以下の値でない場合。
        ValueError: dのshapeが(len(target_indices),)でない場合。
    """
    assert 0 < sigma < 1, "sigma must be in (0, 1)"
    assert 0 < tau < 1, "tau must be in (0, 1)"
    assert alpha > alpha_min, f"alpha must be greater than alpha_min={alpha_min}, but got alpha={alpha}"
    assert alpha_min >= 0., f"alpha_min must be greater than 0, but got alpha_min={alpha_min}"

    if target_indices is None:
        target_indices = tuple(range(x.shape[0]))

    if d is None:
        d = -grad_f_current.clone() #最急降下方向
    assert d.shape == (len(target_indices),), f"d must have shape ({len(target_indices)},), but got {d.shape}"

    with torch.no_grad():
        while alpha > alpha_min:
            x_new = x.clone()
            x_new[target_indices,] = x[target_indices,] + alpha * d
            if func(x_new) <= f_current + sigma * alpha * torch.dot(grad_f_current, d):
                return alpha
            alpha *= tau

    return alpha_min


def line_wolfe(
        func:"Function",
        x:torch.Tensor,
        tau:float,
        alpha:float,
        f_current:torch.float,
        grad_f_current:torch.Tensor,
        sigma:float = 0.5,
        beta:float = 0.9,
        alpha_min:float = 0.,
        d:torch.Tensor = None,
        target_indices:tuple[int] = None,)->float:
    """バックトラッキング法によるWolfe条件を満たす実数alphaを出力する。

    dを降下方向としたととき、入力xはx + alpha*dに更新される。

    Args:
        func (Function): 目的関数。出力はスカラー値である必要がある。
        x (torch.Tensor): 現時点の候補解。shapeは(変数の次元数,)である必要がある。
        f_current (torch.float): 現時点の候補解xにおける目的関数の値。
        grad_f_current (torch.Tensor): 現時点の候補解xにおける目的関数の勾配。shapeは(len(target_indices),)である必要がある。
        target_indices (tuple[int]): 入力のうち更新対象のインデックスのタプル。例えば、xが3次元でtarget_indices=(0, 2)とすると、x[0]とx[2]のみ最適化対象。
        sigma (float): Armijo条件のパラメータ。0 < sigma < 1である必要がある。
        beta (float): Wolfe条件のパラメータ。sigma < beta < 1である必要がある。
        tau (float): バックトラッキング法のパラメータ。0 < tau < 1である必要がある。
        alpha (float): 初期ステップサイズ。alpha_min < alphaである必要がある。
        alpha_min (float, optional): ステップサイズの下限。
        d (torch.Tensor, optional): 降下方向。shapeは(len(target_indices),)である必要がある。Noneの場合はfuncの最急降下方向とする。
    Returns:
        float: Wolfe条件を満たすステップサイズalpha。alpha_minより大きいalphaで条件を満たすものが見つからなかった場合はalpha_minを返す。
    Raises:
        ValueError: sigmaが0以上1以下の値でない場合。
        ValueError: betaがsigma以上1以下の値でない場合。
        ValueError: tauが0以上1以下の値でない場合。
        ValueError: dのshapeが(len(target_indices),)でない場合。
    """
    assert 0 < sigma < beta < 1, "sigma and beta must satisfy 0 < sigma < beta < 1"
    assert 0 < tau < 1, "tau must be in (0, 1)"
    assert alpha > alpha_min, f"alpha must be greater than alpha_min={alpha_min}, but got alpha={alpha}"
    assert alpha_min >= 0., f"alpha_min must be greater than 0, but got alpha_min={alpha_min}"

    if target_indices is None:
        target_indices = tuple(range(x.shape[0]))

    if d is None:
        d = -grad_f_current.clone()  # 最急降下方向
    assert d.shape == (len(target_indices),), f"d must have shape ({len(target_indices)},), but got {d.shape}"

    grad_f_current_dot_d = torch.dot(grad_f_current, d)

    while alpha > alpha_min:
        x_new = x.detach().clone()
        x_new[target_indices,] = x[target_indices,] + alpha * d
        x_new.requires_grad_(True)

        f_new = func(x_new)
        grad_f_new = torch.autograd.grad(f_new, x_new)[0][target_indices,]

        if f_new <= f_current + sigma * alpha * grad_f_current_dot_d and torch.dot(grad_f_new, d) >= beta * grad_f_current_dot_d:
            return alpha

        alpha *= tau

    return alpha_min


def line_strong_wolfe(
        func:"Function",
        x:torch.Tensor,
        tau:float,
        alpha:float,
        f_current:torch.float,
        grad_f_current:torch.Tensor,
        sigma:float = 0.5,
        beta:float = 0.9,
        alpha_min:float = 0.,
        d:torch.Tensor = None,
        target_indices:tuple[int] = None,)->float:
    """バックトラッキング法による強Wolfe条件を満たす実数alphaを出力する。

    dを降下方向としたととき、入力xはx + alpha*dに更新される。

    Args:
        func (Function): 目的関数。出力はスカラー値である必要がある。
        x (torch.Tensor): 現時点の候補解。shapeは(変数の次元数,)である必要がある。
        f_current (torch.float): 現時点の候補解xにおける目的関数の値。
        grad_f_current (torch.Tensor): 現時点の候補解xにおける目的関数の勾配。shapeは(len(target_indices),)である必要がある。
        target_indices (tuple[int]): 入力のうち更新対象のインデックスのタプル。例えば、xが3次元でtarget_indices=(0, 2)とすると、x[0]とx[2]のみ最適化対象。
        sigma (float): Armijo条件のパラメータ。0 < sigma < 1である必要がある。
        beta (float): 強Wolfe条件のパラメータ。sigma < beta < 1である必要がある。
        tau (float): バックトラッキング法のパラメータ。0 < tau < 1である必要がある。
        alpha (float): 初期ステップサイズ。alpha_min < alphaである必要がある。
        alpha_min (float, optional): ステップサイズの下限。
        d (torch.Tensor, optional): 降下方向。shapeは(len(target_indices),)である必要がある。Noneの場合はfuncの最急降下方向とする。
    Returns:
        float: 強Wolfe条件を満たすステップサイズalpha。alpha_minより大きいalphaで条件を満たすものが見つからなかった場合はalpha_minを返す。
    Raises:
        ValueError: sigmaが0以上1以下の値でない場合。
        ValueError: betaがsigma以上1以下の値でない場合。
        ValueError: tauが0以上1以下の値でない場合。
        ValueError: dのshapeが(len(target_indices),)でない場合。
    """
    assert 0 < sigma < beta < 1, "sigma and beta must satisfy 0 < sigma < beta < 1"
    assert 0 < tau < 1, "tau must be in (0, 1)"
    assert alpha > alpha_min, f"alpha must be greater than alpha_min={alpha_min}, but got alpha={alpha}"
    assert alpha_min >= 0., f"alpha_min must be greater than 0, but got alpha_min={alpha_min}"

    if target_indices is None:
        target_indices = tuple(range(x.shape[0]))

    if d is None:
        d = -grad_f_current.clone()  # 最急降下方向
    assert d.shape == (len(target_indices),), f"d must have shape ({len(target_indices)},), but got {d.shape}"

    grad_f_current_dot_d = torch.dot(grad_f_current, d)
    abs_grad_f_current_dot_d = torch.abs(grad_f_current_dot_d)

    while alpha > alpha_min:
        x_new = x.detach().clone()
        x_new[target_indices,] = x[target_indices,] + alpha * d
        x_new.requires_grad_(True)

        f_new = func(x_new)
        grad_f_new = torch.autograd.grad(f_new, x_new)[0][target_indices,]

        if f_new <= f_current + sigma * alpha * grad_f_current_dot_d and torch.abs(torch.dot(grad_f_new, d)) <= beta * abs_grad_f_current_dot_d:
            return alpha

        alpha *= tau

    return alpha_min

