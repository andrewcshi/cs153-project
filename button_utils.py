import discord
from discord.ext import commands
from typing import List, Callable, Awaitable, Optional, Dict, Any, Union

class ButtonView(discord.ui.View):
    """A view that contains buttons for quick replies."""
    
    def __init__(self, *, timeout: Optional[float] = 180):
        super().__init__(timeout=timeout)
        self.value = None
        self.interaction = None
    
    async def on_timeout(self) -> None:
        """Called when the view times out."""
        for item in self.children:
            item.disabled = True
        
        if self.interaction:
            try:
                # Try to edit the message using the interaction
                await self.interaction.response.edit_message(view=self)
            except:
                # If that fails, the interaction might have already been responded to
                try:
                    await self.interaction.message.edit(view=self)
                except:
                    # If all else fails, just pass
                    pass

def create_buttons(options: List[str], 
                   callback: Optional[Callable[[discord.Interaction, str], Awaitable[None]]] = None,
                   *, 
                   timeout: Optional[float] = 180,
                   placeholder: Optional[str] = None) -> ButtonView:
    """
    Create a view with buttons for quick replies.
    
    Args:
        options: List of button labels
        callback: Optional callback function to call when a button is pressed
        timeout: Timeout for the view
        placeholder: Placeholder text for the view
        
    Returns:
        A ButtonView with the specified buttons
    """
    view = ButtonView(timeout=timeout)
    
    # Define styles to cycle through for visual variety
    styles = [
        discord.ButtonStyle.primary,
    ]
    
    for i, option in enumerate(options):
        # Cycle through button styles
        style = styles[i % len(styles)]
        
        # Create a button with the option as label
        button = discord.ui.Button(label=option, style=style, custom_id=f"button_{i}")
        
        # We need to create a closure to capture the current value of option and button
        def make_callback(button, option):
            async def button_callback(interaction):
                view.value = option
                view.interaction = interaction
                
                # Disable all buttons after selection
                for child in view.children:
                    child.disabled = True
                
                # Update the message to show the selection
                await interaction.response.edit_message(
                    content=f"{interaction.message.content}\n\n**Selected: {option}**", 
                    view=view
                )
                
                # Call the provided callback if any
                if callback:
                    await callback(interaction, option)
            return button_callback
        
        # Set the callback for this button
        button.callback = make_callback(button, option)
        view.add_item(button)
    
    return view

async def send_buttons_message(ctx_or_message: Union[commands.Context, discord.Message, discord.TextChannel], 
                              content: str, 
                              options: List[str],
                              callback: Optional[Callable[[discord.Interaction, str], Awaitable[None]]] = None) -> discord.Message:
    """
    Send a message with buttons for quick replies.
    
    Args:
        ctx_or_message: The context, message, or channel to send to
        content: The content of the message
        options: List of button labels
        callback: Optional callback function to call when a button is pressed
        
    Returns:
        The sent message
    """
    view = create_buttons(options, callback)
    
    if isinstance(ctx_or_message, commands.Context):
        return await ctx_or_message.send(content, view=view)
    elif isinstance(ctx_or_message, discord.TextChannel):
        return await ctx_or_message.send(content, view=view)
    else:
        return await ctx_or_message.reply(content, view=view) 