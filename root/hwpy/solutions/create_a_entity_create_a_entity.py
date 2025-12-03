# Generated solution from SOA run
def create_a_entity(*args, **kwargs):
    """create a entity student with email fiel
    """
    validated = validate_entity_args(*args, **kwargs)
    entity = build_entity(validated)
    entity_with_email = ensure_email_field(entity)
    return entity_with_email