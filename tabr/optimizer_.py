
def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            LinearEmbeddings,
            PeriodicEmbeddings,
        ),
    )

def make_parameter_groups(
    model: nn.Module,
    zero_weight_decay_condition,
    custom_groups: dict[tuple[str], dict],  # [(fullnames, options), ...]
) -> list[dict[str, Any]]:
    custom_fullnames = set()
    custom_fullnames.update(*custom_groups)
    assert sum(map(len, custom_groups)) == len(
        custom_fullnames
    ), 'Custom parameter groups must not intersect'

    parameters_info = {}  # fullname -> (parameter, needs_wd)
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters():
            fullname = f'{module_name}.{name}' if module_name else name
            parameters_info.setdefault(fullname, (parameter, []))[1].append(
                not zero_weight_decay_condition(module_name, module, name, parameter)
            )
    parameters_info = {k: (v[0], all(v[1])) for k, v in parameters_info.items()}

    params_with_wd = {'params': []}
    params_without_wd = {'params': [], 'weight_decay': 0.0}
    custom_params = {k: {'params': []} | v for k, v in custom_groups.items()}

    for fullname, (parameter, needs_wd) in parameters_info.items():
        for fullnames, group in custom_params.items():
            if fullname in fullnames:
                custom_fullnames.remove(fullname)
                group['params'].append(parameter)
                break
        else:
            (params_with_wd if needs_wd else params_with_wd)['params'].append(parameter)
    assert (
        not custom_fullnames
    ), f'Some of the custom parameters were not found in the model: {custom_fullnames}'
    return [params_with_wd, params_without_wd] + list(custom_params.values())


def make_optimizer(
    module: nn.Module,
    type: str,
    *,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_parameter_groups: Optional[dict[tuple[str], dict]] = None,
    **optimizer_kwargs,
) -> torch.optim.Optimizer:
    if custom_parameter_groups is None:
        custom_parameter_groups = {}
    Optimizer = getattr(optim, type)
    parameter_groups = make_parameter_groups(module, zero_weight_decay_condition, custom_parameter_groups)
    return Optimizer(parameter_groups, **optimizer_kwargs)
