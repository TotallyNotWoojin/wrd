import os
from typing import Any
from typing import Dict

from pydantic import BaseModel
from pydantic import Extra
from pydantic import model_validator


class Interface(BaseModel):
    gr: Any = None
    interface: Any = None

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validate that api key and python package exists in environment.

        This function checks if the `gradio` Python package is installed in the environment. If the package is not found, it raises a `ValueError`
        with an appropriate error message.

        Args:
            cls (object): The class to which this method belongs.
            values (Dict): A dictionary containing the environment values.
        Return:
            Dict: The updated `values` dictionary with the `gradio` package imported.
        Raise:
            ValueError: If the `gradio` package is not found in the environment.

        """

        try:
            import gradio as gr

            values["gr"] = gr
        except ImportError:
            raise ValueError(
                "Could not import gradio python package. "
                "Please install it with `pip install gradio`."
            )
        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def prepare_interface(
        self,
        respond,
        reset,
        upload_meta,
        available_tasks,
        share=True,
    ):
        """
        Prepare the Gradio interface for the chatbot.

        This method sets up the Gradio interface for the chatbot.
        It creates various UI components such as a textbox for user input, a checkbox for enabling/disabling chat history,
        a dropdown for selecting tasks, and a clear button to reset the interface. The interface is then launched and stored
        in the `self.interface` attribute.

        Args:
            self (object): The instance of the class.
            respond (function): The function to handle user input and generate responses.
            reset (function): The function to reset the chatbot state.
            upload_meta (Any): meta data.
            available_tasks (list, optional): A list of available tasks. Defaults to an empty list.
            share (bool, optional): Flag indicating whether to enable sharing the interface. Defaults to False.
        Return:
            None

        """

        def submit_api_keys(openai_api_key, serp_api_key):
            # Set environment variables
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["SEPR_API_KEY"] = serp_api_key

            print("keys submitted")
            print(openai_api_key)

        with self.gr.Blocks() as demo:
            chatbot = self.gr.Chatbot(bubble_full_width=False)
            with self.gr.Row():
                msg = self.gr.Textbox(
                    scale=9,
                    label="Question",
                    info="Put your query here and press enter.",
                )
                btn = self.gr.UploadButton(
                    "📁",
                    scale=1,
                    file_types=["image", "video", "audio", "text"],
                )
                check_box = self.gr.Checkbox(
                    scale=1,
                    value=True,
                    label="Use History",
                    info="If checked, the chat history will be sent over along with the next query.",
                )

            with self.gr.Row():
                tasks = self.gr.Dropdown(
                    value=[],
                    choices=available_tasks,
                    multiselect=True,
                    label="Tasks List",
                    info="The list of available tasks. Select the ones that you want to use.",
                )

            with self.gr.Row():
                openai_api_key_input = self.gr.Textbox(
                    label="OpenAI API Key",
                    info="Enter your OpenAI API key here.",
                )
                serp_api_key_input = self.gr.Textbox(
                    label="Serp API Key",
                    info="Enter your Serp API key here.",
                )

            clear = self.gr.ClearButton([msg, chatbot])
            clear.click(reset)

            msg.submit(
                respond,
                [
                    msg,
                    openai_api_key_input,
                    serp_api_key_input,
                    chatbot,
                    check_box,
                    tasks,
                ],
                [msg, chatbot],
            )

            btn.upload(
                upload_meta, [chatbot, btn], [chatbot], queue=False
            )

        demo.launch(share=share)
        self.interface = demo

    def close(self):
        """
        Close the Gradio interface.

        This method closes the Gradio interface associated with the chatbot.
        It calls the `close` method of the interface object stored in the `self.interface` attribute.

        Args:
            self (object): The instance of the class.
        Return:
            None
        """

        self.interface.close()
