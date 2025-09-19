#!/usr/bin/env python3

from affine import quixand as qs

def shop_for_product(sandbox):
    """Complete shopping task demonstration"""
    print("=== AgentGym WebShop Environment Example ===\n")

    # Create environment instance
    print("Creating WebShop environment...")
    env_id = sandbox.proxy.create()
    print(f"Environment created with ID: {env_id}\n")
    
    # Get shopping task instruction
    instruction = sandbox.proxy.instruction_text(env_idx=env_id)
    print(f"Shopping Task: {instruction}\n")
    print("-" * 60)
    
    # Start shopping session
    print("\nStarting at WebShop homepage...")
    observation = sandbox.proxy.observation(env_idx=env_id)
    print(f"   Page: {observation[:150]}...")
    
    # Check available actions
    actions = sandbox.proxy.available_actions(env_idx=env_id)
    print(f"   Available: Search bar: {actions['has_search_bar']}, Items: {len(actions['clickables'])}")
    
    # Search for product based on instruction
    print("\nSearching for products...")
    # Extract key terms from instruction (simplified example)
    search_query = "blue wireless headphones"  # In real usage, parse from instruction
    print(f"   Query: {search_query}")
    
    response = sandbox.proxy.step(
        env_idx=env_id,
        action=f"search[{search_query}]"
    )
    print(f"   Results loaded. Reward: {response['reward']}")
    
    # View search results
    print("\nBrowsing search results...")
    observation = sandbox.proxy.observation(env_idx=env_id)
    print(f"   Page shows: {observation[:200]}...")
    
    actions = sandbox.proxy.available_actions(env_idx=env_id)
    products = [item for item in actions['clickables'] if item.startswith('item_')][:3]
    print(f"   Found {len(products)} products")
    
    if products:
        # Click on first product to view details
        product = products[0]
        print(f"\nViewing product details...")
        print(f"   Clicking: {product}")
        
        response = sandbox.proxy.step(
            env_idx=env_id,
            action=f"click[{product}]"
        )
        print(f"   Product page loaded. Reward: {response['reward']}")
        
        # Check product details
        observation = sandbox.proxy.observation(env_idx=env_id)
        print(f"   Details: {observation[:250]}...")
        
        # Check for buy button
        actions = sandbox.proxy.available_actions(env_idx=env_id)
        buy_buttons = [btn for btn in actions['clickables'] if 'buy' in btn.lower()]
        
        if buy_buttons:
            print(f"\nCompleting purchase...")
            print(f"   Clicking: {buy_buttons[0]}")
            
            response = sandbox.proxy.step(
                env_idx=env_id,
                action=f"click[{buy_buttons[0]}]"
            )
            
            if response['done']:
                print(f"\nTask completed successfully!")
                print(f"   Final reward: {response['reward']}")
            else:
                print(f"   Action performed. Reward: {response['reward']}")
        else:
            print("\n   No buy button found, might need to select options first")
            options = actions['clickables'][:5]
            print(f"   Available options: {options}")
    
    print("\n" + "="*60)
    state = sandbox.proxy.state(env_idx=env_id)
    print(f"Final state: {state}")
    print(f"Task complete: {response.get('done', False)}")
    print(f"Total reward: {response.get('reward', 0)}")
    
    return response.get('reward', 0)


def main():
    print("Building WebShop environment image...")
    image = qs.Templates.agentgym("webshop")

    print("Starting sandbox container...")
    sandbox = qs.Sandbox(template=image)
    print(f"Container ID: {sandbox.container_id[:12]}\n")

    try:
        reward = shop_for_product(sandbox)

        print("\n" + "="*60)
        print(f"Example completed! reward: {reward}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nCleaning up...")
        sandbox.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()