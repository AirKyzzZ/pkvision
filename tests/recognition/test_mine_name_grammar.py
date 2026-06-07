from scripts.mine_name_grammar import token_associations


def test_token_associations_purity():
    rows = [
        {"name": "gainer flip", "physics": {"direction": "backward"}},
        {"name": "gainer full", "physics": {"direction": "backward"}},
        {"name": "front gainer", "physics": {"direction": "forward"}},
    ]
    assoc = token_associations(rows)
    g = assoc["gainer"]["direction"]
    assert g["backward"] == 2 and g["forward"] == 1
    assert assoc["gainer"]["_support"] == 3
