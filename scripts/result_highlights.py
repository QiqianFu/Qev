"""Choose emphasized scores among the models displayed in a comparison."""

COMPARISON_GROUPS = (
    ('qev_9b', 'kev_9b'),
    ('qev_4b', 'kev_4b', 'jevany_4b_pointer', 'jevany_4b_direct'),
    ('qev_2b',),
    ('qev_0p8b', 'nanojev_0p6b'),
)


def highlighted_models(scores, displayed):
    visible = set(displayed)
    winners = set()
    for group in COMPARISON_GROUPS:
        peers = [model for model in group if model in visible and model in scores]
        if not peers:
            continue
        values = {model: round(100 * scores[model], 2) for model in peers}
        best = max(values.values())
        winners.update(model for model, value in values.items() if value == best)
    return winners
