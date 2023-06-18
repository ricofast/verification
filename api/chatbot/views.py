from django.shortcuts import render
from .serializers import ChatGenerateSerializer
from .models import Chat
from .utils import random_string
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from rest_framework.response import Response
from rest_framework import status
import re
import json
# Create your views here.
# Load JSON data
def load_json(file):
    with open(file) as bot_responses:
        print(f"Loaded '{file}' successfully!")
        return json.load(bot_responses)


# Store JSON data
filename = rf"media/json_messages/bot.json"
response_data = load_json(filename)


class ChatGenerateView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    # parser_classes = (FileUploadParser,)

    @csrf_exempt
    def post(self, request, *args, **kwargs):
        chat_serializer = ChatGenerateSerializer(data=request.data)
        updated = 'Question added'
        if chat_serializer.is_valid():
            userid = chat_serializer.data['user']
            newquestion = chat_serializer.data['question']
            newresponse = get_response(newquestion)
            obj, created = Chat.objects.update_or_create(
                user=userid,
                defaults={'question': newquestion, 'response': newresponse},
            )

            return Response(newresponse, status=status.HTTP_201_CREATED)
        else:
            return Response(chat_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def get_response(input_string):
    split_message = re.split(r'\s+|[,;?!.-]\s*', input_string.lower())
    score_list = []

    # Check all the responses
    for response in response_data:
        response_score = 0
        required_score = 0
        required_words = response["required_words"]

        # Check if there are any required words
        if required_words:
            for word in split_message:
                if word in required_words:
                    required_score += 1

        # Amount of required words should match the required score
        if required_score == len(required_words):
            # print(required_score == len(required_words))
            # Check each word the user has typed
            for word in split_message:
                # If the word is in the response, add to the score
                if word in response["user_input"]:
                    response_score += 1

        # Add score to list
        score_list.append(response_score)
        # Debugging: Find the best phrase
        # print(response_score, response["user_input"])

    # Find the best response and return it if they're not all 0
    best_response = max(score_list)
    response_index = score_list.index(best_response)

    # Check if input is empty
    if input_string == "":
        return "Please type something so we can chat :("

    # If there is no good response, return a random one.
    if best_response != 0:
        return response_data[response_index]["bot_response"]

    return random_string()