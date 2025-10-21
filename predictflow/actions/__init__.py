from predictflow.engine.registry import register_action

@register_action
def say_hello(context):
    print("Hello from PredictFlow!")
    return {"greeting": "Hello"}

@register_action
def verify_identity(context):
    print("Verifying identity...")
    # pretend we did some check here
    return {"identity_verified": True}
