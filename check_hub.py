try:
    import langchainhub
    print(f"langchainhub dir: {dir(langchainhub)}")
    if hasattr(langchainhub, 'pull'):
        print("langchainhub has 'pull'")
    else:
        print("langchainhub does NOT have 'pull'")
        
    from langchainhub import Client
    print(f"Client dir: {dir(Client)}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
